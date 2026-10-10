import cv2
import numpy as np
import time
import threading
import os
from ultralytics import YOLO


# ============================================================
# CONFIG - MAX AGGRESSIVE CENTER TRACKING & SMOOTH ZOOM
# ============================================================

DETECT_EVERY = 1
YOLO_FPS = 15                 

SWITCH_COOLDOWN = 0.2         
MIN_HOLD_TIME = 2.0           

DETECT_SIZE = (960, 540)
DISPLAY_SIZE = (640, 360)

# BALANCED DIGITAL ZOOM SETTINGS
TARGET_ZOOM_FACTOR = 1.20  
SMOOTH_FACTOR = 0.06      


# ============================================================
# DETECTION CONFIDENCE
# ============================================================

NORMAL_CONF = 0.15


# ============================================================
# STRICT CENTER MARGINS
# ============================================================

POSITION_SWITCH_MARGIN = 0.05
CURRENT_POSITION_LIMIT = 0.55


# ============================================================
# OUTPUT SPEED & PATHS
# ============================================================

OUTPUT_SLOW_FACTOR = 1.3
OUTPUT_DIR = "output"
BEST_CAM_OUTPUT = os.path.join(OUTPUT_DIR, "best_camera.avi")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CUSTOM MODEL & CAPTURES
# ============================================================

model = YOLO("runs/detect/runs/football/finetune_v2/weights/best.pt")

caps = [
    cv2.VideoCapture("input_videos/md.mp4"),
    cv2.VideoCapture("input_videos/lf.mp4"),
    cv2.VideoCapture("input_videos/rt.mp4")
]

if not all(cap.isOpened() for cap in caps):
    print("[ERROR] One or more videos could not be opened.")
    exit()

source_fps = caps[0].get(cv2.CAP_PROP_FPS)
if source_fps <= 0:
    source_fps = 30.0

FRAME_DELAY = 1.0 / source_fps
OUTPUT_FPS = source_fps / OUTPUT_SLOW_FACTOR


# ============================================================
# SHARED STATE
# ============================================================

latest_frames = [None] * len(caps)
last_boxes = [None] * len(caps)

active_cam = 0 
last_switch_time = time.time()

lock = threading.Lock()
running = True
best_writer = None

# SMOOTH ZOOM TRACKING STATE
smooth_ptz = [[DETECT_SIZE[0] / 2.0, DETECT_SIZE[1] / 2.0, 1.0] for _ in range(len(caps))]


# ============================================================
# CENTER POSITION SCORE
# ============================================================

def calculate_position_score(box, frame_width, frame_height):
    if box is None:
        return 0.0

    x1, y1, x2, y2 = box

    ball_cx = (x1 + x2) / 2.0
    ball_cy = (y1 + y2) / 2.0

    nx = ball_cx / frame_width
    ny = ball_cy / frame_height

    dx = abs(nx - 0.5) / 0.5
    dy = abs(ny - 0.5) / 0.5

    distance = np.sqrt(dx * dx + dy * dy) / np.sqrt(2)
    position_score = 1.0 - distance

    return float(np.clip(position_score, 0.0, 1.0))


# ============================================================
# SMART & EDGE-SAFE CROP & ZOOM FUNCTION
# ============================================================

def apply_smooth_ptz(frame, box, cam_idx):
    h, w = frame.shape[:2]
    curr_cx, curr_cy, curr_zoom = smooth_ptz[cam_idx]

    if box is not None:
        x1, y1, x2, y2 = box
        raw_target_cx = (x1 + x2) / 2.0
        raw_target_cy = (y1 + y2) / 2.0
        
        dist_from_center = np.hypot(raw_target_cx - w / 2.0, raw_target_cy - h / 2.0)
        
        # Edge Detection
        edge_margin_x = w * 0.15
        edge_margin_y = h * 0.15

        is_near_edge = (
            raw_target_cx < edge_margin_x or raw_target_cx > (w - edge_margin_x) or
            raw_target_cy < edge_margin_y or raw_target_cy > (h - edge_margin_y)
        )

        if is_near_edge:
            target_zoom = 1.02
            target_cx = w / 2.0
            target_cy = h / 2.0
        elif dist_from_center < 40:
            target_cx = w / 2.0
            target_cy = h / 2.0
            target_zoom = 1.08
        else:
            target_cx = raw_target_cx
            target_cy = raw_target_cy
            target_zoom = TARGET_ZOOM_FACTOR
    else:
        target_cx = w / 2.0
        target_cy = h / 2.0
        target_zoom = 1.0

    # Smooth Interpolation
    new_cx = curr_cx + (target_cx - curr_cx) * SMOOTH_FACTOR
    new_cy = curr_cy + (target_cy - curr_cy) * SMOOTH_FACTOR
    new_zoom = curr_zoom + (target_zoom - curr_zoom) * SMOOTH_FACTOR

    smooth_ptz[cam_idx] = [new_cx, new_cy, new_zoom]

    crop_w = int(w / new_zoom)
    crop_h = int(h / new_zoom)

    xmin = int(np.clip(new_cx - crop_w / 2, 0, w - crop_w))
    ymin = int(np.clip(new_cy - crop_h / 2, 0, h - crop_h))

    cropped = frame[ymin : ymin + crop_h, xmin : xmin + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ============================================================
# CAMERA READER THREAD
# ============================================================

def camera_reader(idx, cap):
    global running
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / (fps if fps > 0 else 30.0)

    while running:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        frame = cv2.resize(frame, DETECT_SIZE)

        with lock:
            latest_frames[idx] = frame

        elapsed = time.time() - t_start
        if delay - elapsed > 0:
            time.sleep(delay - elapsed)


# ============================================================
# YOLO THREAD
# ============================================================

def yolo_worker():
    global running
    yolo_delay = 1.0 / YOLO_FPS

    while running:
        start = time.time()

        with lock:
            frames = latest_frames.copy()

        for i, frame in enumerate(frames):
            if frame is None:
                continue

            results = model.predict(
                frame,
                conf=NORMAL_CONF,
                imgsz=640,
                verbose=False
            )

            boxes = results[0].boxes
            best_box = None
            best_confidence = 0.0

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) != 0:
                        continue

                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    if best_box is None or confidence > best_confidence:
                        best_box = (x1, y1, x2, y2)
                        best_confidence = confidence

            # বল যদি এই ফ্রেমে না পাওয়া যায়, তবে স্পষ্টভাবে None করে দেওয়া (ফ্যান্টম সুইচ বন্ধের জন্য)
            with lock:
                last_boxes[i] = best_box

        elapsed = time.time() - start
        if yolo_delay - elapsed > 0:
            time.sleep(yolo_delay - elapsed)


# ============================================================
# THREAD START
# ============================================================

for i, cap in enumerate(caps):
    threading.Thread(target=camera_reader, args=(i, cap), daemon=True).start()

threading.Thread(target=yolo_worker, daemon=True).start()


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while running:
        loop_start = time.time()

        with lock:
            boxes = last_boxes.copy()
            frames_raw = latest_frames.copy()

        scores = [
            calculate_position_score(boxes[i], DETECT_SIZE[0], DETECT_SIZE[1])
            for i in range(len(caps))
        ]

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        current_score = scores[active_cam]
        now = time.time()

        # --------------------------------------------------------
        # PHANTOM SWITCHING FIX
        # --------------------------------------------------------
        # ১. শুধুমাত্র যদি অন্তত একটি ক্যামেরায় সত্যি বল থাকে (best_score > 0)
        # ২. নতুন ক্যামেরার স্কোর বর্তমান ক্যামেরার চেয়ে ভালো হতে হবে
        if best_score > 0 and best_idx != active_cam and (now - last_switch_time > MIN_HOLD_TIME):
            switch_condition = False

            if best_score > current_score + POSITION_SWITCH_MARGIN:
                switch_condition = True

            if current_score < CURRENT_POSITION_LIMIT and best_score > current_score:
                switch_condition = True

            if switch_condition:
                print(f"[FIXED SWITCH] CAM {active_cam + 1} -> CAM {best_idx + 1} | Current Score: {current_score:.2f} -> Best Score: {best_score:.2f}")
                active_cam = best_idx
                last_switch_time = now

        # --------------------------------------------------------
        # RENDER WITH DIGITAL ZOOM (960x540)
        # --------------------------------------------------------
        full_res_frames = []
        display_frames = []

        for i in range(len(caps)):
            frame = frames_raw[i]
            if frame is None:
                zoomed_frame = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8)
            else:
                zoomed_frame = apply_smooth_ptz(frame, boxes[i], i)

            # Keep 960x540 for Saving
            full_res_frames.append(zoomed_frame)
            # Resize for Display Output Window
            display_frames.append(cv2.resize(zoomed_frame, DISPLAY_SIZE))

        # DISPLAY FRAME
        display_frame = display_frames[active_cam].copy()
        cv2.putText(display_frame, f"CAM: {active_cam + 1} (Subtle Edge-Safe Zoom)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Best Center Stream", display_frame)

        # --------------------------------------------------------
        # SAVE ORIGINAL 960x540 VIDEO
        # --------------------------------------------------------
        best_save_frame = full_res_frames[active_cam]

        cv2.putText(
            best_save_frame, 
            f"CAM {active_cam + 1}", 
            (20, 45),                     # পজিশন (X, Y)
            cv2.FONT_HERSHEY_SIMPLEX,     # ফন্ট
            1.2,                          # সাইজ
            (0, 255, 0),                  # কালার (সবুজ)
            3                             # থিকনেস
        )

        if best_writer is None:
            h, w = best_save_frame.shape[:2]

            best_writer = cv2.VideoWriter(
                BEST_CAM_OUTPUT,
                cv2.VideoWriter_fourcc(*"XVID"),
                OUTPUT_FPS,
                (w, h)
            )

            print(f"[INFO] Writer started -> {BEST_CAM_OUTPUT} ({w}x{h})")

        best_writer.write(best_save_frame)

        # FRAME TIMING CONTROL
        elapsed = time.time() - loop_start
        sleep_time = FRAME_DELAY - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            running = False
            break

finally:
    # ============================================================
    # CLEANUP & SAVE
    # ============================================================
    running = False
    for cap in caps:
        cap.release()
    if best_writer:
        best_writer.release()
        print(f"[INFO] Saved -> {BEST_CAM_OUTPUT}")
    cv2.destroyAllWindows()
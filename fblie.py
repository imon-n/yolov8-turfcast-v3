import cv2
import numpy as np
import time
import threading
import os
import subprocess
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
# FACEBOOK LIVE
# ============================================================
# IMPORTANT: Load the stream key from an environment variable instead
# of hardcoding it in the script. This avoids accidentally leaking it
# via version control, screenshots, or shared files.
#
# Set it before running, e.g.:
#   export FB_STREAM_KEY="your-key-here"        (Linux/macOS)
#   set FB_STREAM_KEY=your-key-here              (Windows cmd)
#
# If you already shared/committed the old key anywhere, regenerate it
# in Facebook Live Producer before going live.

FACEBOOK_SERVER = "rtmps://live-api-s.facebook.com:443/rtmp/"
FACEBOOK_STREAM_KEY = os.environ.get("FB_STREAM_KEY", "FB-1071062722177000-0-Ab79EsWkh-FBcz80xzTsF3as")

if not FACEBOOK_STREAM_KEY:
    print("[ERROR] FB_STREAM_KEY environment variable is not set.")
    print("        Set it with: export FB_STREAM_KEY=your-key-here")
    exit(1)

FACEBOOK_URL = FACEBOOK_SERVER + FACEBOOK_STREAM_KEY

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

# GOP / keyframe interval: Facebook recommends a keyframe roughly
# every 2 seconds.
GOP_SIZE = max(1, int(round(source_fps * 2)))

# ============================================================
# START FFMPEG -> FACEBOOK LIVE
# ============================================================

print("[INFO] Starting FFmpeg...")
print("[INFO] Connecting to Facebook Live...")

ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "info",
    # ---- video input (raw frames piped from Python) ----
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]}",
    "-r", str(source_fps),
    "-i", "-",
    # ---- silent audio input ----
    # Facebook Live frequently rejects or silently fails on
    # video-only streams. A silent AAC track fixes this reliably.
    "-f", "lavfi",
    "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    # ---- video encoding ----
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-b:v", "2500k",
    "-maxrate", "2500k",
    "-bufsize", "5000k",
    "-g", str(GOP_SIZE),
    "-keyint_min", str(GOP_SIZE),
    "-sc_threshold", "0",
    # ---- audio encoding ----
    "-c:a", "aac",
    "-b:a", "128k",
    "-ar", "44100",
    # ---- stop encoding once the (shorter) video input ends ----
    "-shortest",
    "-f", "flv",
    FACEBOOK_URL
]

ffmpeg_process = subprocess.Popen(
    ffmpeg_cmd,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    bufsize=10 ** 8
)


def ffmpeg_log_reader(proc):
    """Continuously read and print ffmpeg's stderr so connection
    errors (auth failure, RTMPS handshake issues, bitrate problems,
    etc.) are visible instead of failing silently."""
    try:
        for line in iter(proc.stderr.readline, b""):
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[FFMPEG] {text}")
    except Exception as e:
        print("[ERROR] ffmpeg log reader crashed:", e)


ffmpeg_log_thread = threading.Thread(
    target=ffmpeg_log_reader, args=(ffmpeg_process,), daemon=True
)
ffmpeg_log_thread.start()

# Give ffmpeg a moment to attempt the RTMPS handshake and fail fast
# if the connection can't be established at all (bad key, blocked
# port, DNS issue, etc.) before we start burning frames into a dead
# pipe.
time.sleep(2.0)
if ffmpeg_process.poll() is not None:
    print(f"[ERROR] FFmpeg exited immediately with code {ffmpeg_process.returncode}.")
    print("        Check the [FFMPEG] log lines above for the real reason")
    print("        (common causes: invalid/expired stream key, blocked port 443,")
    print("        ffmpeg built without rtmps/openssl support).")
    exit(1)

# ============================================================
# SHARED STATE
# ============================================================

latest_frames = [None] * len(caps)
last_boxes = [None] * len(caps)

active_cam = 0
last_switch_time = time.time()

lock = threading.Lock()
running = True

# ============================================================
# SMOOTH ZOOM TRACKING STATE
# ============================================================

smooth_ptz = [
    [DETECT_SIZE[0] / 2.0, DETECT_SIZE[1] / 2.0, 1.0]
    for _ in range(len(caps))
]

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

        dist_from_center = np.hypot(
            raw_target_cx - w / 2.0,
            raw_target_cy - h / 2.0
        )

        edge_margin_x = w * 0.15
        edge_margin_y = h * 0.15

        is_near_edge = (
            raw_target_cx < edge_margin_x
            or raw_target_cx > (w - edge_margin_x)
            or raw_target_cy < edge_margin_y
            or raw_target_cy > (h - edge_margin_y)
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

    new_cx = curr_cx + (target_cx - curr_cx) * SMOOTH_FACTOR
    new_cy = curr_cy + (target_cy - curr_cy) * SMOOTH_FACTOR
    new_zoom = curr_zoom + (target_zoom - curr_zoom) * SMOOTH_FACTOR

    smooth_ptz[cam_idx] = [new_cx, new_cy, new_zoom]

    crop_w = int(w / new_zoom)
    crop_h = int(h / new_zoom)

    xmin = int(np.clip(new_cx - crop_w / 2, 0, w - crop_w))
    ymin = int(np.clip(new_cy - crop_h / 2, 0, h - crop_h))

    cropped = frame[ymin:ymin + crop_h, xmin:xmin + crop_w]

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
            frames = [f.copy() if f is not None else None for f in latest_frames]

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
            boxes = [b for b in last_boxes]
            frames_raw = [f.copy() if f is not None else None for f in latest_frames]

        scores = [
            calculate_position_score(boxes[i], DETECT_SIZE[0], DETECT_SIZE[1])
            for i in range(len(caps))
        ]

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        current_score = scores[active_cam]
        now = time.time()

        if (
            best_score > 0
            and best_idx != active_cam
            and (now - last_switch_time > MIN_HOLD_TIME)
        ):
            switch_condition = False

            if best_score > current_score + POSITION_SWITCH_MARGIN:
                switch_condition = True

            if current_score < CURRENT_POSITION_LIMIT and best_score > current_score:
                switch_condition = True

            if switch_condition:
                print(
                    f"[SWITCH] CAM {active_cam + 1} -> CAM {best_idx + 1} | "
                    f"{current_score:.2f} -> {best_score:.2f}"
                )
                active_cam = best_idx
                last_switch_time = now

        display_frames = []
        for i in range(len(caps)):
            frame = frames_raw[i]
            if frame is None:
                zoomed_frame = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), np.uint8)
            else:
                zoomed_frame = apply_smooth_ptz(frame, boxes[i], i)

            display_frame = cv2.resize(zoomed_frame, DISPLAY_SIZE)
            display_frames.append(display_frame)

        display_frame = display_frames[active_cam]

        try:
            ffmpeg_process.stdin.write(display_frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            print("[ERROR] FFmpeg/Facebook connection lost:", e)
            print("        Check the [FFMPEG] log lines above for the reason.")
            running = False
            break

        # If ffmpeg died on its own (e.g. Facebook dropped the
        # connection), stop the loop instead of writing into a dead pipe.
        if ffmpeg_process.poll() is not None:
            print(f"[ERROR] FFmpeg process exited unexpectedly (code {ffmpeg_process.returncode}).")
            running = False
            break

        elapsed = time.time() - loop_start
        sleep_time = FRAME_DELAY - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ============================================================
# CLEANUP
# ============================================================

finally:
    running = False
    print("[INFO] Stopping cameras...")

    for cap in caps:
        cap.release()

    if ffmpeg_process:
        try:
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
        except Exception:
            pass

        try:
            ffmpeg_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()

    print("[INFO] Facebook Live stopped.")
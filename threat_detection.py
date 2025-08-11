# threat_detection_integrated.py
import cv2
from ultralytics import YOLO
import serial
import threading
import pyttsx3
import time
from collections import deque
import math
import sys
import os

# ---------------- CONFIG ----------------
SERIAL_PORT = 'COM7'                       # <-- set your ESP32 serial port
BAUD_RATE = 115200

# put your trained knife model here:
KNIFE_MODEL_PATH = r"runs\detect\train2\weights\best.pt"   # <-- update if different
PERSON_MODEL_PATH = "yolov8n.pt"                          # pretrained person detector

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 0, 50            # ESP32 limits (your sketch uses 0..80)

SMOOTHING_WINDOW = 5
ANGLE_THRESHOLD = 3                   # degrees threshold to send updates
TRACK_TIMEOUT = 3.0                   # seconds to keep tracking after last sighting
ASSOCIATE_DISTANCE_THRESHOLD = 150    # px: associate knife -> person if nearer than this

# physical offsets (pixels) — tune to your build:
OFFSET_X_PIXELS = -20   # positive moves aiming to right relative to image
OFFSET_Y_PIXELS = 10    # positive moves aiming down relative to image

# If camera preview is mirrored (left/right reversed) set True.
# This flips the X mapping so the servo moves the same direction in the real world.
FLIP_HORIZONTAL = True

# Optional throttle to reduce CPU usage:
INFERENCE_DELAY = 0.0  # seconds between frames

# ----------------------------------------

# Serial init
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[OK] Opened serial port {SERIAL_PORT} @ {BAUD_RATE}")
except Exception as e:
    print(f"[WARN] Could not open serial {SERIAL_PORT}: {e}")
    ser = None

# TTS init
engine = pyttsx3.init()
alert_running = False
alert_lock = threading.Lock()

def voice_alert_loop():
    """TTS runs in background while alert_running True."""
    while True:
        with alert_lock:
            if not alert_running:
                break
        engine.say("Drop your weapon down. Hands up.")
        engine.runAndWait()
        time.sleep(0.8)

alert_thread = None

def start_alert():
    global alert_running, alert_thread
    with alert_lock:
        if not alert_running:
            alert_running = True
            alert_thread = threading.Thread(target=voice_alert_loop, daemon=True)
            alert_thread.start()

def stop_alert():
    global alert_running
    with alert_lock:
        alert_running = False

# Utility functions
def map_angle(value, in_min, in_max, out_min, out_max):
    """Linearly map value from [in_min,in_max] to [out_min,out_max] and clamp."""
    if in_max == in_min:
        return int((out_min + out_max) / 2)
    v = max(min(value, in_max), in_min)
    return int((v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# Load models
print("Loading models...")
if not os.path.exists(KNIFE_MODEL_PATH):
    print(f"[ERROR] Knife model not found at {KNIFE_MODEL_PATH}. Update KNIFE_MODEL_PATH and retry.")
    sys.exit(1)

# Load models (Ultralytics YOLO)
knife_model = YOLO(KNIFE_MODEL_PATH)    # your custom knife model
person_model = YOLO(PERSON_MODEL_PATH)  # COCO pretrained for person detection
print("Models loaded.")

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit(1)

pan_history = deque(maxlen=SMOOTHING_WINDOW)
tilt_history = deque(maxlen=SMOOTHING_WINDOW)
last_pan = None
last_tilt = None

tracked_person = None   # bbox (x1,y1,x2,y2)
last_seen_time = 0

print("Starting live detection. Press ESC to quit.")

try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Optionally mirror frame for display (does not affect model input)
        # We will compute servo mapping with FLIP_HORIZONTAL variable instead of mirroring input.
        persons = []
        knives = []

        # 1) detect persons (COCO pretrained)
        res_person = person_model(frame)[0]
        for det in res_person.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls_id = det
            cls_id = int(cls_id)
            label = person_model.names.get(cls_id, str(cls_id))
            if label == 'person':
                bbox = (int(x1), int(y1), int(x2), int(y2))
                persons.append((bbox, float(score)))

        # 2) detect knives with your custom model
        res_knife = knife_model(frame)[0]
        for det in res_knife.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, cls_id = det
            cls_id = int(cls_id)
            label = knife_model.names.get(cls_id, str(cls_id))
            bbox = (int(x1), int(y1), int(x2), int(y2))
            knives.append((bbox, float(score), label))

        # Visualization copy
        vis = frame.copy()
        # draw persons
        for (bbox, score) in persons:
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(vis, f"person {score:.2f}", (bbox[0], bbox[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # draw knives
        for (bbox, score, label) in knives:
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
            cv2.putText(vis, f"{label} {score:.2f}", (bbox[0], bbox[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Associate nearest knife -> person (pick the person closest to any knife center)
        threat_person = None
        min_dist = float('inf')
        for (knife_bbox, kscore, klabel) in knives:
            k_cx = (knife_bbox[0] + knife_bbox[2]) / 2
            k_cy = (knife_bbox[1] + knife_bbox[3]) / 2
            for (p_bbox, pscore) in persons:
                p_cx = (p_bbox[0] + p_bbox[2]) / 2
                p_cy = (p_bbox[1] + p_bbox[3]) / 2
                d = euclidean((k_cx, k_cy), (p_cx, p_cy))
                if d < min_dist:
                    min_dist = d
                    threat_person = p_bbox

        # Tracking logic
        current_time = time.time()
        if threat_person is not None and min_dist < ASSOCIATE_DISTANCE_THRESHOLD:
            tracked_person = threat_person
            last_seen_time = current_time
            start_alert()
        else:
            # attempt to keep tracking last person (if still in frame)
            if tracked_person is not None:
                t_cx = (tracked_person[0] + tracked_person[2]) / 2
                t_cy = (tracked_person[1] + tracked_person[3]) / 2
                min_p_dist = float('inf')
                closest = None
                for (p_bbox, pscore) in persons:
                    p_cx = (p_bbox[0] + p_bbox[2]) / 2
                    p_cy = (p_bbox[1] + p_bbox[3]) / 2
                    d = euclidean((t_cx, t_cy), (p_cx, p_cy))
                    if d < min_p_dist:
                        min_p_dist = d
                        closest = p_bbox
                if closest is not None and min_p_dist < ASSOCIATE_DISTANCE_THRESHOLD:
                    tracked_person = closest
                    last_seen_time = current_time
                else:
                    if current_time - last_seen_time > TRACK_TIMEOUT:
                        tracked_person = None
                        stop_alert()

        # Aim and send to ESP32 if tracking
        if tracked_person is not None:
            # forehead = center-x and near top of bbox (tune fraction as needed)
            forehead_x = int((tracked_person[0] + tracked_person[2]) / 2)
            forehead_y = int(tracked_person[1] + 0.15 * (tracked_person[3] - tracked_person[1]))

            # apply offset of laser vs camera
            fx = forehead_x + OFFSET_X_PIXELS
            fy = forehead_y + OFFSET_Y_PIXELS

            # clamp
            fx = max(0, min(FRAME_WIDTH, fx))
            fy = max(0, min(FRAME_HEIGHT, fy))

            # account for mirrored preview: servo_x must correspond to real-world left/right
            servo_x = FRAME_WIDTH - fx if FLIP_HORIZONTAL else fx

            # map to servo angles (tilt inverted)
            pan_angle = map_angle(servo_x, 0, FRAME_WIDTH, PAN_MIN, PAN_MAX)
            tilt_angle = map_angle(fy, 0, FRAME_HEIGHT, TILT_MAX, TILT_MIN)

            # smoothing
            pan_history.append(pan_angle)
            tilt_history.append(tilt_angle)
            smoothed_pan = int(sum(pan_history)/len(pan_history))
            smoothed_tilt = int(sum(tilt_history)/len(tilt_history))

            # send only if changed more than threshold to avoid serial spam
            if (last_pan is None or abs(smoothed_pan - last_pan) > ANGLE_THRESHOLD or
                last_tilt is None or abs(smoothed_tilt - last_tilt) > ANGLE_THRESHOLD):
                cmd = f"{smoothed_pan},{smoothed_tilt},1\n"
                if ser and ser.is_open:
                    try:
                        ser.write(cmd.encode())
                    except Exception as e:
                        print(f"[WARN] Serial write failed: {e}")
                last_pan = smoothed_pan
                last_tilt = smoothed_tilt

            # visualize where laser is aimed in camera view (account for the display mirroring logic)
            vis_x = fx if not FLIP_HORIZONTAL else FRAME_WIDTH - fx
            cv2.circle(vis, (int(vis_x), int(fy)), 10, (0,255,255), 3)
        else:
            # no tracked person -> ensure laser off
            if ser and ser.is_open:
                try:
                    ser.write(b"0,0,0\n")
                except Exception:
                    pass
            last_pan = None
            last_tilt = None
            pan_history.clear()
            tilt_history.clear()

        # Show frame (flip horizontally for a natural selfie-like view if you prefer)
        # If you want the preview mirrored, uncomment the next line:
        # vis = cv2.flip(vis, 1)
        cv2.imshow("Threat Detection (knife+person)", vis)

        # exit
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed - exiting")
            break

        if INFERENCE_DELAY > 0:
            time.sleep(INFERENCE_DELAY)

finally:
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
    stop_alert()
    time.sleep(0.5)
    print("Program terminated.")
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime
import requests

# ================= CONFIG =================
MODEL_PATH = "violence_model_epoch_02.tflite"

FRAME_WIDTH = 128
FRAME_HEIGHT = 128
SEQUENCE_LENGTH = 20
SKIP_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.30

VIOLENCE_CONFIRM_SECONDS = 2.0
CAPTURE_INTERVAL_SECONDS = 1.0
MAX_CAPTURES = 2
INCIDENT_COOLDOWN_SECONDS = 0

CAPTURE_DIR = "captures"
API_URL = "https://resq-server.onrender.com/api/violence-detected/"
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "AI-VISION-SURVEILLANCE-01"

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ================= API =================
def send_violence_event(image_paths, confidence_score):
    files = []
    for path in image_paths:
        files.append(
            ("images", (os.path.basename(path), open(path, "rb"), "image/jpeg"))
        )

    data = {
        "beacon_id": BEACON_ID,
        "confidence_score": f"{confidence_score:.2f}",
        "description": "Confirmed violent incident – auto captured sequence",
        "device_id": DEVICE_ID,
    }

    try:
        r = requests.post(API_URL, data=data, files=files, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("❌ API error:", e)
        return None

# ================= MODEL =================
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found:", MODEL_PATH)
    exit()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(sequence):
    x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][1]

# ================= STATE =================
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

violence_start_time = None
violence_confirmed = False
last_capture_time = 0
capture_count = 0
captured_images = []
last_incident_time = 0

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Webcam not available")
    exit()

print("📹 Running violence detection — press Q to quit")

frame_count = 0
current_label = "Initializing..."
current_color = (255, 255, 255)

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frames_queue.append(resized / 255.0)

    frame_count += 1

    if len(frames_queue) == SEQUENCE_LENGTH and frame_count % SKIP_FRAMES == 0:
        score = predict(frames_queue)

        if score > CONFIDENCE_THRESHOLD:
            current_label = f"⚠️ VIOLENCE ({score:.1%})"
            current_color = (0, 0, 255)

            if violence_start_time is None:
                violence_start_time = time.time()
            elif time.time() - violence_start_time >= VIOLENCE_CONFIRM_SECONDS:
                violence_confirmed = True
        else:
            current_label = f"🟢 Safe ({1-score:.1%})"
            current_color = (0, 255, 0)

            violence_start_time = None
            violence_confirmed = False
            capture_count = 0
            captured_images.clear()

    # ========== CAPTURE ==========
    if violence_confirmed and capture_count < MAX_CAPTURES:
        now = time.time()
        if now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{CAPTURE_DIR}/violence_{ts}_{capture_count+1}.jpg"

            cv2.imwrite(path, frame)
            captured_images.append(path)
            capture_count += 1
            last_capture_time = now

            print("📸 Captured:", path)

            if capture_count == MAX_CAPTURES:
                if now - last_incident_time >= INCIDENT_COOLDOWN_SECONDS:
                    print("🚨 Sending incident to backend...")
                    result = send_violence_event(captured_images, score)
                    if result:
                        incident_id = result.get("incident_id")
                        status = result.get("status", "unknown")

                        if incident_id:
                            print(f"✅ Incident created successfully | ID: {incident_id}")
                            last_incident_time = now
                        else:
                            print("⚠️ Incident response received but no incident_id")
                            print("🔎 Full response:", result)


                # HARD RESET
                violence_confirmed = False
                violence_start_time = None
                capture_count = 0
                captured_images.clear()

    # ========== UI ==========
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, current_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

    cv2.imshow("Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

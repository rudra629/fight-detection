import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime


# --- CONFIGURATION ---
MODEL_PATH = 'violence_model_epoch_02.tflite'
SKIP_FRAMES = 5  # Run AI only every 5th frame to stop lag

# --- CAPTURE CONFIG ---
VIOLENCE_CONFIRM_SECONDS = 2.0
CAPTURE_INTERVAL_SECONDS = 1.0
MAX_CAPTURES = 3
CAPTURE_DIR = "captures"

violence_start_time = None
violence_confirmed = False

last_capture_time = 0
capture_count = 0

os.makedirs(CAPTURE_DIR, exist_ok=True)


if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Could not find model file: {MODEL_PATH}")
    exit()

print(f"Loading TFLite Model: {MODEL_PATH}...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Settings
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.30
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

def predict_frame(sequence):
    input_data = np.array(sequence, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][1]

# --- WEBCAM SETUP ---
# Try 0, 1, or the DroidCam URL if needed
cap = cv2.VideoCapture(0)

# OPTIMIZATION 1: Lower Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# OPTIMIZATION 2: Limit Buffer (Windows/Linux support varies, but good to have)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📹 Starting smooth webcam... Press 'q' to quit.")

frame_count = 0
current_label = "Initializing..."
current_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret: break

    # OPTIMIZATION 3: Resize logic
    # We only resize for the AI, but show the original 'frame' to user
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    normalized = resized / 255.0
    frames_queue.append(normalized)

    # OPTIMIZATION 4: Skip Frames
    # Only run the heavy prediction every 'SKIP_FRAMES' times
    frame_count += 1

    if len(frames_queue) == SEQUENCE_LENGTH and frame_count % SKIP_FRAMES == 0:
        score = predict_frame(frames_queue)

        if score > CONFIDENCE_THRESHOLD:
            current_label = f"⚠️ VIOLENCE ({score:.1%})"
            current_color = (0, 0, 255)

            # --- TIME-BASED CONFIRMATION ---
            if violence_start_time is None:
                violence_start_time = time.time()

            elif time.time() - violence_start_time >= VIOLENCE_CONFIRM_SECONDS:
                violence_confirmed = True

        else:
            current_label = f"🟢 Safe ({1-score:.1%})"
            current_color = (0, 255, 0)

            # Reset everything
            violence_start_time = None
            violence_confirmed = False
            capture_count = 0

    # --- CAPTURE LOGIC ---
    if violence_confirmed and capture_count < MAX_CAPTURES:
        now = time.time()

        if now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{CAPTURE_DIR}/violence_{timestamp}_{capture_count+1}.jpg"

            cv2.imwrite(filename, frame)
            print(f"📸 Captured: {filename}")

            last_capture_time = now
            capture_count += 1

            if capture_count == MAX_CAPTURES:
                print("✅ Capture sequence completed")


    # Draw the LAST KNOWN label (so it doesn't flicker)
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, current_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

    cv2.imshow('Violence Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
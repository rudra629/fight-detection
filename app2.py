import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime
import requests
import tempfile

# ================= CONFIG =================
# Ensure this matches your model filename exactly
MODEL_PATH = "violence_model_epoch_02_NEW.tflite"

# Detection Constants
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.50  # Adjusted slightly for stability

# Logic Constants (Your Custom Logic)
VIOLENCE_CONFIRM_SECONDS = 2.0  # Violence must persist this long to trigger alert
CAPTURE_INTERVAL_SECONDS = 1.0  # Time between taking evidence photos
MAX_CAPTURES = 2                # How many photos to send per incident
INCIDENT_COOLDOWN_SECONDS = 10  # Don't spam the server (wait 10s between incidents)

# API Constants
API_URL = "https://resq-server.onrender.com/api/violence-detected/"
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "STREAMLIT-CLIENT-01"

# Setup Directory
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ================= STATE MANAGEMENT =================
# We use st.session_state to keep variables alive between Streamlit reruns
if 'frames_queue' not in st.session_state:
    st.session_state.frames_queue = deque(maxlen=SEQUENCE_LENGTH)
if 'violence_start_time' not in st.session_state:
    st.session_state.violence_start_time = None
if 'violence_confirmed' not in st.session_state:
    st.session_state.violence_confirmed = False
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'last_incident_time' not in st.session_state:
    st.session_state.last_incident_time = 0

# ================= API FUNCTION =================
def send_violence_event(image_paths, confidence_score):
    """Sends the captured evidence to the backend API."""
    files = []
    open_files = []
    try:
        for path in image_paths:
            f = open(path, "rb")
            open_files.append(f)
            files.append(
                ("images", (os.path.basename(path), f, "image/jpeg"))
            )

        data = {
            "beacon_id": BEACON_ID,
            "confidence_score": f"{confidence_score:.2f}",
            "description": "Streamlit Detected Incident",
            "device_id": DEVICE_ID,
        }

        with st.spinner("🚨 Violence Confirmed! Sending Alert..."):
            r = requests.post(API_URL, data=data, files=files, timeout=15)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None
    finally:
        # Close all file handles safely
        for f in open_files:
            f.close()

# ================= MODEL LOADER =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None, None, None
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

def predict(sequence):
    """Runs inference on a sequence of frames."""
    if interpreter is None:
        return 0.0
    
    # Prepare input data [1, 20, 128, 128, 3]
    x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
    
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    
    # Get output (Handling different model output shapes)
    output_data = interpreter.get_tensor(output_details[0]["index"])
    if output_data.shape[-1] > 1:
        # If output is [Safe_Prob, Violence_Prob]
        return output_data[0][1] 
    else:
        # If output is single sigmoid probability
        return output_data[0][0]

# ================= MAIN UI =================
st.title("🛡️ AI Violence Detection System")
st.markdown("---")

st.sidebar.header("Configuration")
input_source = st.sidebar.radio("Select Input Source:", ("Webcam (Live)", "Upload Video File"))

# Placeholders for video and alerts
stframe = st.empty()
alert_box = st.empty()

def process_frame(frame_bgr):
    """Core logic: runs detection, confirmation timer, and API triggering."""
    
    # 1. Preprocess
    # Convert BGR (OpenCV) to RGB (Model expects RGB usually)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize and Normalize
    resized = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT))
    normalized = resized / 255.0
    
    # Add to Queue
    st.session_state.frames_queue.append(normalized)

    # UI Defaults
    label = "Initializing..."
    color = (0, 255, 255) # Yellow
    score = 0.0

    # 2. Prediction Logic (Only if queue is full)
    if len(st.session_state.frames_queue) == SEQUENCE_LENGTH:
        score = predict(st.session_state.frames_queue)

        # CHECK VIOLENCE
        if score > CONFIDENCE_THRESHOLD:
            label = f"⚠️ VIOLENCE ({score:.1%})"
            color = (255, 0, 0) # Red

            # Timer Logic (Your 2-second Confirmation)
            if st.session_state.violence_start_time is None:
                st.session_state.violence_start_time = time.time()
            elif time.time() - st.session_state.violence_start_time >= VIOLENCE_CONFIRM_SECONDS:
                st.session_state.violence_confirmed = True
        
        # CHECK SAFE
        else:
            label = f"🟢 Safe ({1-score:.1%})"
            color = (0, 255, 0) # Green
            
            # Reset triggers if violence stops
            st.session_state.violence_start_time = None
            st.session_state.violence_confirmed = False
            st.session_state.capture_count = 0
            st.session_state.captured_images.clear()

    # 3. Capture & API Logic
    if st.session_state.violence_confirmed and st.session_state.capture_count < MAX_CAPTURES:
        now = time.time()
        
        # Check cooldown from previous incident
        if now - st.session_state.last_incident_time > INCIDENT_COOLDOWN_SECONDS:
            
            if now - st.session_state.last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                # Save Image
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"violence_{ts}_{st.session_state.capture_count+1}.jpg"
                path = os.path.join(CAPTURE_DIR, filename)
                
                # Save the original BGR frame
                cv2.imwrite(path, frame_bgr)
                
                st.session_state.captured_images.append(path)
                st.session_state.capture_count += 1
                st.session_state.last_capture_time = now
                
                alert_box.warning(f"📸 Capturing Evidence {st.session_state.capture_count}/{MAX_CAPTURES}...")

                # If we have enough images, SEND ALERT
                if st.session_state.capture_count == MAX_CAPTURES:
                    result = send_violence_event(st.session_state.captured_images, score)
                    
                    if result:
                        st.success(f"✅ Alert Sent! Incident ID: {result.get('incident_id', 'Unknown')}")
                        st.session_state.last_incident_time = now
                    
                    # Hard Reset
                    st.session_state.violence_confirmed = False
                    st.session_state.violence_start_time = None
                    st.session_state.capture_count = 0
                    st.session_state.captured_images.clear()

    # 4. Draw UI on Frame for Display
    # Draw text banner
    cv2.rectangle(frame_rgb, (0, 0), (600, 50), (0, 0, 0), -1)
    cv2.putText(frame_rgb, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame_rgb

# ================= VIDEO RUNNER =================
if input_source == "Webcam (Live)":
    st.write("Click 'Start' to enable webcam.")
    run = st.checkbox("Start Camera")
    
    if run:
        # Use OpenCV for continuous frame capture
        cap = cv2.VideoCapture(1)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break
            
            # Process & Display
            output_frame = process_frame(frame)
            stframe.image(output_frame, channels="RGB", use_column_width=True)
            
        cap.release()

elif input_source == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.write("Processing Video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process & Display
            output_frame = process_frame(frame)
            stframe.image(output_frame, channels="RGB", use_column_width=True)

        cap.release()

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
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ================= CONFIGURATION =================
MODEL_PATH = "violence_model_epoch_02.tflite"

# Detection Constants
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.50

# Logic Constants
VIOLENCE_CONFIRM_SECONDS = 2.0
CAPTURE_INTERVAL_SECONDS = 1.0
MAX_CAPTURES = 2
INCIDENT_COOLDOWN_SECONDS = 10

# API Constants
API_URL = "https://resq-server.onrender.com/api/violence-detected/"
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "STREAMLIT-CLIENT-01"

CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ================= MODEL LOADER =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None, None, None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

interpreter, input_details, output_details = load_model()

# ================= API HANDLER =================
def run_api_call(image_paths, score):
    """Runs API call in background thread."""
    try:
        files = []
        open_files = []
        for path in image_paths:
            f = open(path, "rb")
            open_files.append(f)
            files.append(("images", (os.path.basename(path), f, "image/jpeg")))

        data = {
            "beacon_id": BEACON_ID,
            "confidence_score": f"{score:.2f}",
            "description": "Streamlit Detected Incident",
            "device_id": DEVICE_ID,
        }
        
        requests.post(API_URL, data=data, files=files, timeout=15)
        print("✅ API Alert Sent Successfully")
    except Exception as e:
        print(f"❌ API Error: {e}")
    finally:
        for f in open_files:
            f.close()

# ================= VIDEO PROCESSOR =================
class ViolenceProcessor(VideoTransformerBase):
    def __init__(self):
        self.frames_queue = deque(maxlen=SEQUENCE_LENGTH)
        self.violence_start_time = None
        self.violence_confirmed = False
        self.last_capture_time = 0
        self.capture_count = 0
        self.captured_images = []
        self.last_incident_time = 0

    def recv(self, frame):
        # 1. Convert WebRTC frame to BGR (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Preprocess for Model (RGB -> Resize -> Normalize)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (FRAME_WIDTH, FRAME_HEIGHT))
        normalized = resized / 255.0
        
        self.frames_queue.append(normalized)

        # Defaults
        label = "Scanning..."
        color = (255, 255, 0) # Cyan
        score = 0.0

        # 3. Prediction Logic
        if len(self.frames_queue) == SEQUENCE_LENGTH:
            if interpreter:
                # Prepare Input
                x = np.expand_dims(np.array(self.frames_queue, dtype=np.float32), axis=0)
                
                # Run Inference
                interpreter.set_tensor(input_details[0]["index"], x)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])
                
                # Parse Output
                if output_data.shape[-1] > 1:
                    score = output_data[0][1]
                else:
                    score = output_data[0][0]

            # 4. Violence Logic
            if score > CONFIDENCE_THRESHOLD:
                label = f"⚠️ VIOLENCE ({score:.1%})"
                color = (0, 0, 255) # Red

                if self.violence_start_time is None:
                    self.violence_start_time = time.time()
                elif time.time() - self.violence_start_time >= VIOLENCE_CONFIRM_SECONDS:
                    self.violence_confirmed = True
            else:
                label = f"🟢 Safe ({1-score:.1%})"
                color = (0, 255, 0) # Green
                self.violence_start_time = None
                self.violence_confirmed = False
                self.capture_count = 0
                self.captured_images = []

        # 5. Capture & Alert Logic
        if self.violence_confirmed and self.capture_count < MAX_CAPTURES:
            now = time.time()
            if now - self.last_incident_time > INCIDENT_COOLDOWN_SECONDS:
                if now - self.last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                    # Save Image
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"violence_{ts}_{self.capture_count+1}.jpg"
                    path = os.path.join(CAPTURE_DIR, filename)
                    cv2.imwrite(path, img) # Save original BGR
                    
                    self.captured_images.append(path)
                    self.capture_count += 1
                    self.last_capture_time = now
                    
                    # On-Screen Feedback
                    cv2.putText(img, f"CAPTURING EVIDENCE {self.capture_count}/{MAX_CAPTURES}", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Send Alert
                    if self.capture_count == MAX_CAPTURES:
                        self.last_incident_time = now
                        t = threading.Thread(target=run_api_call, args=(list(self.captured_images), score))
                        t.start()
                        
                        # Reset
                        self.violence_confirmed = False
                        self.violence_start_time = None
                        self.capture_count = 0
                        self.captured_images = []

        # 6. Draw Overlay
        cv2.rectangle(img, (0, 0), (600, 50), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Return frame to browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= MAIN UI =================
st.set_page_config(page_title="AI Violence Detector", page_icon="🛡️")
st.title("🛡️ AI Violence Detection System")

st.sidebar.header("Settings")
input_source = st.sidebar.radio("Select Input Source:", ("Webcam (Live)", "Upload Video File"))

# --- WEB RTC CONFIGURATION (STUN SERVERS) ---
# This block is crucial for fixing the "Connection taking too long" error
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

if input_source == "Webcam (Live)":
    st.write("Permissions: Allow your browser to access the camera.")
    
    # Run the streamer
    webrtc_streamer(
        key="violence-detection",
        video_processor_factory=ViolenceProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif input_source == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # We manually use the processor class for files
        processor = ViolenceProcessor()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mimic WebRTC flow: Create AV Frame -> Recv -> Convert back
            # (We skip AV conversion for speed here and just use the logic)
            
            # 1. Preprocess logic manual copy
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (FRAME_WIDTH, FRAME_HEIGHT))
            normalized = resized / 255.0
            processor.frames_queue.append(normalized)

            label = "Scanning..."
            color = (0, 255, 0)
            
            if len(processor.frames_queue) == SEQUENCE_LENGTH:
                x = np.expand_dims(np.array(processor.frames_queue, dtype=np.float32), axis=0)
                interpreter.set_tensor(input_details[0]["index"], x)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])
                
                score = output_data[0][1] if output_data.shape[-1] > 1 else output_data[0][0]
                
                if score > CONFIDENCE_THRESHOLD:
                    label = f"⚠️ VIOLENCE ({score:.1%})"
                    color = (0, 0, 255)
                else:
                    label = f"🟢 Safe ({1-score:.1%})"

            # Draw
            cv2.rectangle(frame, (0, 0), (600, 50), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            stframe.image(frame, channels="BGR")
            
        cap.release()

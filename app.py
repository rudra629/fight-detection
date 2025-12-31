import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import tensorflow as tf
import requests
import time
import av
from collections import deque
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="ResQ AI Vision", page_icon="🚨", layout="wide")

MODEL_PATH = 'violence_model_epoch_02.tflite'
API_URL = "https://resq-server.onrender.com/api/violence-detected/"
CONFIDENCE_THRESHOLD = 0.70

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model globally so it's shared
interpreter = load_tflite_model()

# --- VIDEO PROCESSOR CLASS ---
# This class runs INSIDE the video stream thread
class ViolenceDetector(VideoProcessorBase):
    def __init__(self):
        self.frames_queue = deque(maxlen=20) # 20 Frame Sequence
        self.last_alert_time = 0
        self.alert_cooldown = 5 # Seconds
        self.interpreter = interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.prediction = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Get image from stream
        image = frame.to_ndarray(format="bgr24")
        
        # 2. Preprocess
        # Resize to 128x128 for AI
        resized = cv2.resize(image, (128, 128))
        normalized = resized / 255.0
        self.frames_queue.append(normalized)

        # 3. Predict (Only if queue is full)
        if len(self.frames_queue) == 20:
            # Prepare input
            input_data = np.array(self.frames_queue, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            self.prediction = output_data[0][1] # Violence score

            # 4. API Alert Logic
            if self.prediction > CONFIDENCE_THRESHOLD:
                current_time = time.time()
                if (current_time - self.last_alert_time) > self.alert_cooldown:
                    try:
                        # Send POST in background (simple try/except to not block video)
                        payload = {
                            "beacon_id": "ab907856-3412-3412-3412-341278563412",
                            "confidence_score": float(self.prediction),
                            "description": "Fight detected via Cloud AI",
                            "device_id": "STREAMLIT-CLOUD-001"
                        }
                        requests.post(API_URL, json=payload, timeout=1)
                        print(f"🚀 Alert Sent! Score: {self.prediction}")
                        self.last_alert_time = current_time
                    except:
                        pass # Ignore network errors to keep video smooth

        # 5. Draw UI on the frame
        if self.prediction > CONFIDENCE_THRESHOLD:
            # Red Box & Text
            cv2.rectangle(image, (0,0), (640, 50), (0,0,255), -1)
            cv2.putText(image, f"VIOLENCE: {self.prediction:.1%}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Green Text
            cv2.putText(image, f"Safe: {1-self.prediction:.1%}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Return the annotated frame to the browser
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- FRONTEND UI ---
st.title("🚨 ResQ Cloud Vision")
st.write("This app runs entirely in the cloud. Click 'Start' and allow camera access.")

# WebRTC Streamer
# This replaces cv2.VideoCapture
ctx = webrtc_streamer(
    key="violence-detection",
    video_processor_factory=ViolenceDetector,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Status indicators
if ctx.state.playing:
    st.success("✅ Monitoring Active")
    st.info("The AI is processing frames. Violence > 80% triggers the API.")
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import requests
import time
from collections import deque
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Violence Detection AI", page_icon="🚨", layout="wide")

# --- SETTINGS ---
MODEL_PATH = 'violence_model_epoch_02.tflite'
SEQUENCE_LENGTH = 20
FRAME_HEIGHT = 128
FRAME_WIDTH = 128
CONFIDENCE_THRESHOLD = 0.70

# --- SPEED SETTINGS ---
AI_SKIP_FRAMES = 5 
UI_SKIP_FRAMES = 3 

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# --- UI & CONFIGURATION ---
st.title("🚨 ResQ AI Vision Node")

# Sidebar
st.sidebar.header("Configuration")
enable_ai = st.sidebar.checkbox("Enable AI Detection", value=True)

# 1. Update the API URL to your specific endpoint
default_url = "https://resq-server.onrender.com/api/violence-detected/"
api_url = st.sidebar.text_input("API Endpoint", default_url)

cam_source = st.sidebar.radio("Camera Source", ["Webcam (0)", "DroidCam (1)", "IP Camera"])

cam_index = 0
if cam_source == "DroidCam (1)":
    cam_index = 1
elif cam_source == "IP Camera":
    cam_index = st.sidebar.text_input("IP URL", "http://192.168.x.x:4747/video")

col1, col2 = st.columns([2, 1])
with col1:
    st_frame = st.empty()
with col2:
    status_text = st.empty()
    confidence_bar = st.progress(0)
    log_area = st.empty()

def run_app():
    cap = cv2.VideoCapture(cam_index)
    
    # Low resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_counter = 0
    last_score = 0.0
    last_alert_time = 0
    
    # Prevent spamming the server (send max 1 alert every 5 seconds)
    ALERT_COOLDOWN = 5 
    
    start_button = st.sidebar.button("Start System", key="start")
    stop_button = st.sidebar.button("Stop System", key="stop")

    if start_button and not stop_button:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera disconnected.")
                break
            
            frame_counter += 1

            # --- AI LOGIC ---
            if enable_ai and frame_counter % AI_SKIP_FRAMES == 0:
                resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                normalized = resized / 255.0
                frames_queue.append(normalized)

                if len(frames_queue) == SEQUENCE_LENGTH:
                    input_data = np.array(frames_queue, dtype=np.float32)
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    last_score = output_data[0][1]

                    # --- POST REQUEST LOGIC ---
                    if last_score > CONFIDENCE_THRESHOLD:
                        current_time = time.time()
                        
                        if (current_time - last_alert_time) > ALERT_COOLDOWN:
                            try:
                                # 2. Construct the Exact Payload
                                payload = {
                                    "beacon_id": "ab907856-3412-3412-3412-341278563412",
                                    "confidence_score": float(last_score), # Live confidence
                                    "description": "Fight detected between 2 people near library entrance",
                                    "device_id": "AI-VISION-001"
                                }
                                
                                # 3. Send Request
                                response = requests.post(api_url, json=payload, timeout=2)
                                
                                # Log the result
                                if response.status_code == 200 or response.status_code == 201:
                                    log_area.success(f"✅ Alert Sent! (Server: {response.status_code})")
                                else:
                                    log_area.error(f"❌ Server Error: {response.status_code}")
                                    
                                last_alert_time = current_time
                                
                            except Exception as e:
                                log_area.warning(f"⚠️ Connection Failed: {e}")

            # --- UI LOGIC ---
            if frame_counter % UI_SKIP_FRAMES == 0:
                color = (0, 255, 0)
                label = f"Safe ({last_score:.0%})"
                
                if last_score > CONFIDENCE_THRESHOLD:
                    color = (255, 0, 0)
                    label = f"VIOLENCE ({last_score:.0%})"
                
                # Draw Box
                cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
                cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                status_text.markdown(f"**Current Status:** {label}")
                confidence_bar.progress(int(last_score * 100))

                # Resize for browser
                frame_small = cv2.resize(frame, (320, 240)) 
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

if __name__ == "__main__":
    run_app()
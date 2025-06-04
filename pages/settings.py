import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="Settings - SmartFlow", page_icon="⚙️")

st.title("System Settings")
st.markdown("Configure system parameters and preferences")

# Camera Settings
st.subheader("Camera Settings")
camera_settings = st.expander("Camera Configuration", expanded=True)
with camera_settings:
    camera_source = st.selectbox("Camera Source", ["IP Camera", "USB Camera", "Video File"])
    if camera_source == "IP Camera":
        ip_address = st.text_input("IP Address", "192.168.1.100")
        port = st.number_input("Port", 8000, 9000, 8080)
    elif camera_source == "USB Camera":
        device_id = st.number_input("Device ID", 0, 10, 0)
    else:
        video_path = st.text_input("Video File Path", "data/raw/sample.mp4")
    
    resolution = st.selectbox("Resolution", ["1080p", "720p", "480p"])
    fps = st.slider("Frames per Second", 1, 30, 15)

# AI Model Settings
st.subheader("AI Model Settings")
model_settings = st.expander("Model Configuration", expanded=True)
with model_settings:
    model_type = st.selectbox("Model Type", ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    detection_classes = st.multiselect(
        "Detection Classes",
        ["Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrian"],
        default=["Car", "Truck", "Bus"]
    )

# Traffic Signal Settings
st.subheader("Traffic Signal Settings")
signal_settings = st.expander("Signal Configuration", expanded=True)
with signal_settings:
    min_green_time = st.number_input("Minimum Green Time (seconds)", 10, 60, 20)
    max_red_time = st.number_input("Maximum Red Time (seconds)", 30, 120, 60)
    yellow_time = st.number_input("Yellow Time (seconds)", 3, 10, 5)
    
    optimization_mode = st.selectbox(
        "Optimization Mode",
        ["Traffic Flow", "Pedestrian Priority", "Emergency Vehicle Priority", "Balanced"]
    )

# System Settings
st.subheader("System Settings")
system_settings = st.expander("System Configuration", expanded=True)
with system_settings:
    data_storage = st.selectbox("Data Storage Location", ["Local", "Cloud"])
    if data_storage == "Cloud":
        cloud_provider = st.selectbox("Cloud Provider", ["AWS", "Azure", "Google Cloud"])
        st.text_input("API Key", type="password")
    
    log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    auto_update = st.checkbox("Enable Automatic Updates", value=True)

# Save Settings
if st.button("Save Settings"):
    st.success("Settings saved successfully!")
    # Placeholder for settings save functionality 
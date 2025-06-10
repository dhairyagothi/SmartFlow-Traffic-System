import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import logging
import tempfile
import serial
import serial.tools.list_ports
import os
import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartTraffic")

# Page configuration
st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="wide")

# Display current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"**Current Date and Time:** {current_time}")
st.markdown("**Current User's Login:** dhairyagothi")

# Initialize session state variables
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = {
        'lane1': {'vehicles': 0, 'file': None, 'file_type': None},
        'lane2': {'vehicles': 0, 'file': None, 'file_type': None},
        'lane3': {'vehicles': 0, 'file': None, 'file_type': None},
        'lane4': {'vehicles': 0, 'file': None, 'file_type': None}
    }

if 'priority_lane' not in st.session_state:
    st.session_state.priority_lane = None

if 'temp_video_paths' not in st.session_state:
    st.session_state.temp_video_paths = {
        'lane1': None, 'lane2': None, 'lane3': None, 'lane4': None
    }

if 'green_timer_active' not in st.session_state:
    st.session_state.green_timer_active = False

# Classes for vehicle detection (COCO dataset)
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

# Load YOLOv8 model
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        logger.info("YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded video file to temporary location and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        return None

def detect_vehicles(image_array, model):
    """Run vehicle detection on an image and return count and annotated image"""
    try:
        # Run detection
        results = model(image_array, verbose=False)[0]
        
        # Count vehicles
        count = 0
        annotated_img = image_array.copy()
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                count += 1
                # Draw rectangle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return count, annotated_img
        
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        return 0, image_array

def extract_video_frame(video_path):
    """Extract a frame from video for detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        # Read a frame from the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_idx = np.random.randint(0, max(1, int(total_frames/2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        
        cap.release()
        return None
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        return None

def find_arduino_port():
    """Find the Arduino port automatically"""
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description:
            return port.device
    return None

def initialize_arduino():
    """Initialize Arduino connection"""
    try:
        port = find_arduino_port()
        if port:
            arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            logger.info(f"Arduino connected on port {port}")
            return arduino
        else:
            logger.error("No Arduino found")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Arduino: {str(e)}")
        return None

def send_green_signal(arduino, lane):
    """Send command to Arduino to turn on green LED on pin 9"""
    try:
        if arduino and arduino.is_open:
            # Send command to turn on green LED on pin 9
            arduino.write(b'GREEN_ON\n')
            logger.info(f"Sent GREEN signal 🟢 to Arduino for Lane {lane}")
            
            # Start timer to turn off after 10 seconds if not already running
            if not st.session_state.green_timer_active:
                st.session_state.green_timer_active = True
                timer_thread = threading.Thread(target=turn_off_green_after_delay, args=(arduino, 10))
                timer_thread.daemon = True
                timer_thread.start()
            
            return True
    except Exception as e:
        logger.error(f"Error sending command to Arduino: {str(e)}")
    return False

def turn_off_green_after_delay(arduino, delay_seconds):
    """Turn off green LED after specified delay"""
    try:
        time.sleep(delay_seconds)
        if arduino and arduino.is_open:
            arduino.write(b'GREEN_OFF\n')
            logger.info(f"Turned OFF green signal after {delay_seconds} seconds")
        st.session_state.green_timer_active = False
    except Exception as e:
        logger.error(f"Error in timer thread: {str(e)}")
        st.session_state.green_timer_active = False

def analyze_traffic_and_update_priority():
    """Detect vehicles in all lanes and update priority"""
    model = load_model()
    if not model:
        st.error("Failed to load detection model. Please check your installation.")
        return
    
    # Process each lane
    for i in range(4):
        lane_key = f'lane{i+1}'
        lane_data = st.session_state.traffic_data[lane_key]
        file = lane_data['file']
        file_type = lane_data['file_type']
        
        if file:
            try:
                img_array = None
                
                # Get image based on file type
                if file_type == "image":
                    file.seek(0)
                    image = Image.open(file).convert("RGB")
                    img_array = np.array(image)
                elif file_type == "video":
                    video_path = st.session_state.temp_video_paths[lane_key]
                    if video_path:
                        img_array = extract_video_frame(video_path)
                
                if img_array is not None:
                    # Run detection
                    count, annotated_img = detect_vehicles(img_array, model)
                    
                    # Update session state
                    st.session_state.traffic_data[lane_key]['vehicles'] = count
                    st.session_state[f"annotated_image_{i}"] = annotated_img
                    logger.info(f"Lane {i+1}: Detected {count} vehicles")
            except Exception as e:
                logger.error(f"Error processing Lane {i+1}: {str(e)}")
    
    # Find highest priority lane (most vehicles)
    lane_priorities = [(i+1, st.session_state.traffic_data[f'lane{i+1}']['vehicles']) 
                      for i in range(4)]
    lane_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Set priority lane to the one with most vehicles
    if lane_priorities:
        highest_priority_lane = lane_priorities[0][0]
        st.session_state.priority_lane = highest_priority_lane
        logger.info(f"Priority updated: Lane {highest_priority_lane} with {lane_priorities[0][1]} vehicles")
        
        # Send signal to Arduino
        send_green_signal(st.session_state.arduino, highest_priority_lane)

def cleanup_temp_files():
    """Remove temporary video files when app exits"""
    for lane_key, path in st.session_state.temp_video_paths.items():
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.info(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing temporary file {path}: {str(e)}")

# Register cleanup handler
import atexit
atexit.register(cleanup_temp_files)

# Initialize Arduino
if 'arduino' not in st.session_state:
    st.session_state.arduino = initialize_arduino()

# UI COMPONENTS
st.title("🚦 Smart Traffic Management System")

# Main control button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Traffic & Update Signal", type="primary", use_container_width=True)

# Create a container for lane images
with st.container():
    # Create 2x2 grid for lane images
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    
    # Layout for all 4 lanes
    for i, cols in enumerate([row1_cols, row2_cols]):
        for j, col in enumerate(cols):
            lane_idx = i*2 + j
            lane_num = lane_idx + 1
            lane_key = f'lane{lane_num}'
            
            with col:
                st.markdown(f"### Lane {lane_num}")
                
                # File type selection
                file_type = st.radio(f"File Type", 
                                   ["Image", "Video"],
                                   horizontal=True,
                                   key=f"file_type_{lane_num}")
                
                # Upload file based on type
                if file_type == "Image":
                    uploaded_file = st.file_uploader(f"Upload image for Lane {lane_num}", 
                                                type=["jpg", "png", "jpeg"], 
                                                key=f"file_{lane_num}")
                    
                    if uploaded_file:
                        # Store the file in session state
                        st.session_state.traffic_data[lane_key]['file'] = uploaded_file
                        st.session_state.traffic_data[lane_key]['file_type'] = "image"
                        
                        # Display original image
                        image = Image.open(uploaded_file).convert("RGB")
                        st.image(image, caption=f"Lane {lane_num} Image", use_column_width=True)
                else:  # Video
                    uploaded_file = st.file_uploader(f"Upload video for Lane {lane_num}", 
                                                type=["mp4", "avi", "mov", "mkv"], 
                                                key=f"video_{lane_num}")
                    
                    if uploaded_file:
                        # Save video to temp file and store path
                        video_path = save_uploaded_file(uploaded_file)
                        if video_path:
                            st.session_state.traffic_data[lane_key]['file'] = uploaded_file
                            st.session_state.traffic_data[lane_key]['file_type'] = "video"
                            st.session_state.temp_video_paths[lane_key] = video_path
                            
                            # Display video player
                            st.video(video_path)
                            
                            # Extract and show a frame from the video
                            preview_frame = extract_video_frame(video_path)
                            if preview_frame is not None:
                                st.image(
                                    preview_frame,
                                    caption=f"Preview Frame from Lane {lane_num} Video",
                                    use_column_width=True
                                )
                        else:
                            st.error(f"Failed to process video for Lane {lane_num}")

# Detection results area
st.markdown("---")
st.subheader("Traffic Analysis Results")

# Display detection results in a more organized way
st.markdown("""
<style>
.detection-card {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
}
.vehicle-count {
    font-size: 24px;
    font-weight: bold;
    color: #1E88E5;
}
.priority-lane {
    background-color: #174717;
    border-left: 5px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

detection_cols = st.columns(4)
for i in range(4):
    lane_num = i + 1
    lane_key = f'lane{i+1}'
    
    with detection_cols[i]:
        is_priority = st.session_state.priority_lane == lane_num
        card_class = "detection-card priority-lane" if is_priority else "detection-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h3>Lane {lane_num} {' (PRIORITY)' if is_priority else ''}</h3>
            <p>Vehicles: <span class="vehicle-count">{st.session_state.traffic_data[lane_key]['vehicles']}</span></p>
            <p>Status: {'🟢 GREEN' if is_priority else '🔴 RED'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detection results if available
        if f"annotated_image_{i}" in st.session_state and st.session_state[f"annotated_image_{i}"] is not None:
            st.image(
                st.session_state[f"annotated_image_{i}"],
                caption=f"Lane {lane_num}: {st.session_state.traffic_data[lane_key]['vehicles']} vehicles detected",
                use_column_width=True
            )

# Priority status with visualization
st.markdown("---")
st.subheader("Lane Priority Visualization")

# Show priority lane visualization
if st.session_state.priority_lane:
    priority_lane = st.session_state.priority_lane
    
    # Display all lanes in priority order with visual cues
    lane_priorities = [(i+1, st.session_state.traffic_data[f'lane{i+1}']['vehicles']) 
                      for i in range(4)]
    lane_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Create a horizontal bar chart to visualize vehicle counts
    labels = [f"Lane {lane}" for lane, _ in lane_priorities]
    values = [count for _, count in lane_priorities]
    colors = ['green' if lane == priority_lane else 'red' for lane, _ in lane_priorities]
    
    # Display as a table with colored bars
    st.markdown("### Priority Order (by Vehicle Count)")
    
    max_count = max(values) if values else 1
    for rank, (lane, count) in enumerate(lane_priorities, 1):
        # Calculate bar width as percentage of max
        bar_width = int(100 * count / max_count) if max_count > 0 else 0
        bar_color = "#4CAF50" if lane == priority_lane else "#F44336"  # green or red
        
        # Create visual bar with label
        st.markdown(f"""
        <div style="margin-bottom: 10px; display: flex; align-items: center;">
            <div style="width: 80px; text-align: center; margin-right: 10px;">Lane {lane}</div>
            <div style="width: {bar_width}%; background-color: {bar_color}; height: 30px; 
                 display: flex; align-items: center; padding-left: 10px; color: white;">
                {count} vehicles
            </div>
            <div style="margin-left: 10px;">
                {' 🟢 GREEN (Active)' if lane == priority_lane else ' 🔴 RED'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show timer information for green signal
    if st.session_state.green_timer_active:
        st.info("⏱️ Green signal will automatically turn off after 10 seconds")
else:
    st.warning("No lane has been analyzed yet. Click 'Analyze Traffic' to detect vehicles and set priority.")

# Arduino status
st.markdown("---")
st.subheader("Arduino Status")
if st.session_state.arduino and st.session_state.arduino.is_open:
    st.success("✅ Arduino Connected - Ready to send signals to pin 9")
else:
    st.error("❌ Arduino Not Connected - Cannot send signals")

# Handle analyze button click
if analyze_button:
    with st.spinner("Analyzing traffic in all lanes..."):
        analyze_traffic_and_update_priority()
        st.experimental_rerun()  # Force refresh to show updated UI
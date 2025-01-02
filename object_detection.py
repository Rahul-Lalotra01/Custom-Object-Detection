import streamlit as st  
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np

# Load your trained YOLO model
model = YOLO(r"C:\Users\Rahul\Downloads\yolov10l_epoch_211.pt")

# Streamlit UI setup
st.title("Object Detection with YOLO")
st.sidebar.title("Options")

# Sidebar options
input_option = st.sidebar.selectbox("Choose Input Type", ("Webcam", "Image", "Video"))

# Function to perform detection
def perform_detection(image):
    results = model(image)
    return results[0].plot()  # Returns image with bounding boxes and labels

# Webcam Detection
if input_option == "Webcam":
    st.sidebar.write("Webcam will open in a separate window.")
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if not st.session_state.webcam_running:
        if st.button("Start Webcam", key="start_webcam_button"):
            st.session_state.webcam_running = True
    else:
        if st.button("Stop Webcam", key="stop_webcam_button"):
            st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            st.write("Webcam feed will be displayed below.")
            frame_placeholder = st.empty()

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture image from webcam.")
                    break

                annotated_frame = perform_detection(frame)
                frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
        else:
            st.error("Could not access the webcam. Please ensure it is connected and try again.")

# Image Detection
elif input_option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        annotated_image = perform_detection(image_np)
        st.image(annotated_image, caption="Detected Image", use_container_width=True)

# Video Detection
elif input_option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_file.name)

        if cap.isOpened():
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame = perform_detection(frame)
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            cap.release()
        else:
            st.error("Could not process the video. Please try another file.")

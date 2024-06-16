# Python In-built packages
from pathlib import Path
import PIL
from PIL.ExifTags import TAGS, GPSTAGS

# External packages
import streamlit as st
import cv2
import tempfile
import numpy as np
import folium
from streamlit_folium import st_folium

# Local Modules
import settings_images
import helper_images

# Setting page layout
st.set_page_config(
    page_title="Tree Infection Segmentation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Tree Infection Detection/Segmentation")
st.markdown('''**NOTE: This model is used to detect infection tree detection and segmentation**''')

# Logo
st.sidebar.image("Hole_detection.JPG")

# Model Options
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 1, 100, 40)) / 100

model_path = Path(settings_images.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper_images.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Choose Source")
source_radio = st.sidebar.radio(
    "Source Type", settings_images.SOURCES_LIST)

# Buttons for detection
detect_button = st.sidebar.button('Detect Infection')
detect_video_objects_button = st.sidebar.button('Detect Video Objects')

source_video = None
source_imgs = None

# Function to extract coordinates from an image
def get_coordinates(image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]

        if gps_info:
            lat = gps_info.get("GPSLatitude")
            lat_ref = gps_info.get("GPSLatitudeRef")
            lon = gps_info.get("GPSLongitude")
            lon_ref = gps_info.get("GPSLongitudeRef")

            if lat and lon and lat_ref and lon_ref:
                lat = convert_to_degrees(lat)
                if lat_ref != "N":
                    lat = -lat

                lon = convert_to_degrees(lon)
                if lon_ref != "E":
                    lon = -lon

                return lat, lon

        return None, None
    except Exception as e:
        st.error("Error extracting coordinates: {}".format(e))
        return None, None

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

# Function to display a map with a marker
def display_map(latitude, longitude):
    map_location = folium.Map(location=[latitude, longitude], zoom_start=15)
    folium.Marker([latitude, longitude], popup='Your Location').add_to(map_location)
    return map_location

# Function to extract coordinates from a video
def get_video_coordinates(video_path):
    # This is a placeholder function, implement logic to extract GPS data from video
    # For demonstration, let's return some dummy coordinates
    return 37.7749, -122.4194  # Coordinates of San Francisco

# Function to process video and skip frames
def process_video(source_video, model, confidence, skip_frames):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(source_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps / (skip_frames + 1), (width, height))

    stframe = st.empty()

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % (skip_frames + 1) == 0:
            pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = model.predict(pil_img, conf=confidence)
            res_plotted = res[0].plot()[:, :, ::-1]
            out.write(res_plotted)
            stframe.image(res_plotted, channels="BGR")

        frame_number += 1

    cap.release()
    out.release()

# If images are selected
if source_radio == settings_images.IMAGE:
    source_imgs = st.sidebar.file_uploader(
        "Choose images...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), accept_multiple_files=True)

    if source_imgs:
        for source_img in source_imgs:
            col1, col2 = st.columns(2)

            with col1:
                try:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

                    # Extract and display coordinates
                    lat, lon = get_coordinates(uploaded_image)
                    if lat and lon:
                        st.write(f"Coordinates: Latitude {lat}, Longitude {lon}")
                        map_location = display_map(lat, lon)
                        st_folium(map_location, width=700, height=500)
                    else:
                        st.write("No GPS data found in the image.")

                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if detect_button:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write('The array of detected pixels')
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

# If video is selected
elif source_radio == settings_images.VIDEO:
    source_video = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov", "mkv"))

    if source_video is not None:
        skip_frames = st.sidebar.slider('Skip Frames', 0, 40, 2)

        # No map display for videos
        # Extract and display coordinates (you can remove this part if you don't want any GPS data handling for videos)
        lat, lon = get_video_coordinates(source_video)
        if lat and lon:
            st.write(f"Coordinates: Latitude {lat}, Longitude {lon}")

        st.video(source_video)

        if detect_video_objects_button:
            process_video(source_video, model, confidence, skip_frames)

else:
    st.error("Please select a valid source type!")

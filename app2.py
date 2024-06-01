# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settingss
import helper

# Setting page layout
st.set_page_config(
    page_title="Tree Infection Segmentation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Tree Infection Detection/Segmentation")
st.markdown('''**NOTE: This model is  used to detect infection tree detection and segmentation**''')

#Logo
st.sidebar.image("Hole_detection.JPG")


# Model Options
#model_type = st.sidebar.radio(
    #"Task Type:", ['Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 5, 100, 40)) / 100

# # Selecting Detection Or Segmentation
# if model_type == 'Detection':
#     model_path = Path(settings.DETECTION_MODEL)
#if model_type == 'Segmentation':
model_path = Path(settingss.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Choose any Image")
source_radio = st.sidebar.radio(
    "Source Type", settingss.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settingss.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settingss.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settingss.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Infection'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write('The array of detected pixels')
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

# elif source_radio == settings.VIDEO:
#     helper.play_stored_video(confidence, model)

# elif source_radio == settings.WEBCAM:
#     helper.play_webcam(confidence, model)

# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

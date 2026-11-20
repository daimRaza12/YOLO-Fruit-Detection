
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os

st.set_page_config(
    page_title="YOLO Fruit Detection",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 YOLO Fruit Detection System")
st.write("Upload an image to detect different fruit categories.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

if not os.path.exists("best.pt"):
    st.error("Model file best.pt not found!")
    st.stop()

model = load_model()

confidence = st.slider(
    "Adjust Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Convert annotated image from BGR to RGB
    annotated_image = result.plot()[:, :, ::-1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Fruits")
        st.image(annotated_image, use_container_width=True)

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())
            score = float(box.conf.item())

            detections.append({
                "Fruit": model.names[class_id],
                "Confidence (%)": round(score * 100, 2)
            })

    st.subheader("Detection Summary")

    if detections:

        df = pd.DataFrame(detections)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"Total objects detected: {len(detections)}"
        )

    else:
        st.warning("No fruits detected!")

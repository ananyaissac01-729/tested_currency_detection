import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("runs/detect/train3/weights/best.pt")

st.title("Indian Currency Detection")

uploaded_file = st.file_uploader("Upload an image of currency", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]

            note = class_name.split("_")[0]

            st.success(f"The note is ₹{note}")

    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Detection Result")
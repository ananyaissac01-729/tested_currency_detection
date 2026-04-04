from ultralytics import YOLO
from PIL import Image
import streamlit as st

model = YOLO("runs/detect/train2/weights/best.pt")

st.title("Indian Currency Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)

    detected = False

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            if class_name == "not_currency":
                if conf > 0.85:
                    st.error("Not real currency")
                else:
                    note = "Unknown"
            elif conf > 0.8:
                note = class_name.split("_")[0]
                st.success(f"The note is ₹{note}")
            else:
                st.error("Not real currency")
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detection Result")

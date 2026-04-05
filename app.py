from ultralytics import YOLO
from PIL import Image
import streamlit as st
from gtts import gTTS
import os
import base64

# 🚀 Load model once (important for Streamlit Cloud)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("💸 Indian Currency Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)

    note = None  # ✅ initialize safely
    best_conf = 0  # track best detection

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = r.names[class_id]

            # pick BEST detection only
            if conf > best_conf:
                best_conf = conf

                if class_name == "not_currency":
                    note = None
                else:
                    note = class_name.split("_")[0]

    # 🎯 Decision logic
    if best_conf > 0.92 and note:
        st.success(f"The note is ₹{note}")
    else:
        st.error("Not real currency")
        note = None

    # 🖼 Show annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detection Result")

    # 🔊 Voice output (only if valid)


    if note:
        text = f"The detected currency is {note} rupees"
 
        tts = gTTS(text)
        tts.save("output.mp3")

        with open("output.mp3", "rb") as f:
            audio_bytes = f.read()

        b64 = base64.b64encode(audio_bytes).decode()

        st.markdown(
            f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import streamlit as st
import gdown

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🍅 TomatoVision <span style='color:#E53935;'>AI</span></h2>", unsafe_allow_html=True)
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")


if not os.path.exists(MODEL_PATH):
    file_id = "1Z-M2gkm79eRnm8HARBIA_c8Jg82Zw4Xg"
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Model downloading... Please wait 1-2 minutes for first time..."):
        gdown.download(url, MODEL_PATH, quiet=False)

CLASS_NAMES = ["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.write("Upload a Tomato Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        result = CLASS_NAMES[class_idx]
    
    st.success(f"Result: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
    
    if result == "healthy":
        st.balloons()
        st.write("Plant is Healthy")
    else:
        st.warning(f"Disease Detected: {result}")
else:
    st.info("Please upload an image to start prediction")

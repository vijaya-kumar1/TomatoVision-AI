
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem;}
    h2 {text-align: center !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🍅 TomatoVision <span style='color:#E53935;'>AI</span></h2>", unsafe_allow_html=True)
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")

CLASS_NAMES = ["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","JPG"])
    if uploaded:
        img = ImageOps.exif_transpose(Image.open(uploaded).convert("RGB"))
        st.image(img, width=300, caption="Uploaded Tomato Leaf")
        predict_btn = st.button("🔍 Predict Disease", width=300, type="primary")
    else:
        predict_btn = False

with col_result:
    if uploaded and predict_btn:
        model = load_model()
        size = model.input_shape[1]
        arr = np.asarray(img.resize((size, size)), dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        scores = model.predict(arr, verbose=0)[0]
        if not np.isclose(scores.sum(), 1.0, atol=1e-3):
            scores = tf.nn.softmax(scores).numpy()
        idx = np.argmax(scores)
        st.markdown(f"<div style='text-align:center; padding:30px; border-radius:15px; background:#E8F5E9; border:2px solid #4CAF50;'><h2 style='color:#2E7D32;'>🦠 {CLASS_NAMES[idx]}</h2><h3 style='color:#1565C0;'>📊 {scores[idx]*100:.2f}%</h3></div>", unsafe_allow_html=True)

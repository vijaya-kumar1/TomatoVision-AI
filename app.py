import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 32px; font-weight: bold; }
    .red-text { color: #FF3131; }
    .result-box {
        background-color: #E8F5E9; border: 2px solid #66BB6A;
        border-radius: 12px; padding: 40px; text-align: center; margin-top: 20px;
    }
    .error-box {
        background-color: #FFEBEE; border: 2px solid #EF5350;
        border-radius: 12px; padding: 30px; text-align: center; margin-top: 20px;
    }
    .disease-text { color: #2E7D32; font-size: 28px; font-weight: bold; }
    .error-text { color: #C62828; font-size: 20px; font-weight: bold; }
    .confidence-text { color: #1565C0; font-size: 22px; font-weight: bold; margin-top: 15px;}
    .stButton>button { background-color: #FF4B4B; color: white; width: 100%; border-radius: 8px; font-weight: bold; height: 45px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍅 TomatoVision <span class="red-text">AI</span></div>', unsafe_allow_html=True)
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")
CLASS_NAMES = ["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def is_tomato_leaf(image):
    img_small = image.resize((100, 100))
    arr = np.array(img_small)
    white_pixels = np.sum(np.all(arr > 200, axis=2)) / 10000
    if white_pixels > 0.6:
        return False, "This looks like a document/paper, not a leaf! 📄"
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green_dominant = np.sum((g > r+10) & (g > b+10)) / 10000
    if green_dominant < 0.15:
        return False, "No leaf found! Please upload only tomato leaf 🍃"
    return True, "OK"

left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.write("**Choose an image**")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded is not None:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)
        predict_btn = st.button("🔵 Predict Disease")
    else:
        predict_btn = False
        image = None

with right_col:
    if uploaded is not None and predict_btn:
        is_leaf, msg = is_tomato_leaf(image)
        if not is_leaf:
            st.markdown(f'<div class="error-box"><div class="error-text">❌ {msg}<br><br>Upload only tomato leaf photo!</div></div>', unsafe_allow_html=True)
        else:
            model = load_trained_model()
            input_size = int(model.input_shape[1])
            img_resized = image.resize((input_size, input_size))
            image_array = np.asarray(img_resized, dtype=np.float32)[None, ...]
            scores = model.predict(image_array, verbose=0)[0]
            if not np.isclose(scores.sum(), 1.0, atol=1e-3):
                scores = tf.nn.softmax(scores).numpy()
            pred_index = int(np.argmax(scores))
            pred_name = CLASS_NAMES[pred_index]
            confidence = float(scores[pred_index]) * 100
            if confidence < 70:
                st.warning(f"⚠️ Low confidence ({confidence:.2f}%). Image may be unclear.")
            st.markdown(f'<div class="result-box"><div class="disease-text">🦠 {pred_name}</div><div class="confidence-text">📊 {confidence:.2f}%</div></div>', unsafe_allow_html=True)
    elif uploaded is None:
        st.info("👈 Upload image on left side")

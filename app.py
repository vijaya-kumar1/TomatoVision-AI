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
    .result-box { background-color: #E8F5E9; border: 2px solid #66BB6A; border-radius: 12px; padding: 40px; text-align: center; margin-top: 20px; }
    .error-box { background-color: #FFEBEE; border: 2px solid #EF5350; border-radius: 12px; padding: 30px; text-align: center; margin-top: 20px; }
    .disease-text { color: #2E7D32; font-size: 28px; font-weight: bold; }
    .error-text { color: #C62828; font-size: 22px; font-weight: bold; }
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

def is_tomato_leaf_strict(image):
    img_small = image.resize((120, 120))
    arr = np.array(img_small).astype(float)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    max_rgb = np.maximum(np.maximum(r,g), b)
    min_rgb = np.minimum(np.minimum(r,g), b)
    sat = (max_rgb - min_rgb) / (max_rgb + 1e-5)
    
    white_mask = (max_rgb > 195) & (sat < 0.25)
    valid = ~white_mask
    
    
    skin = (r > 95) & (g > 35) & (b > 15) & (r > g+15) & (r > b+10) & (r-g > 15) & (r-g < 85) & valid
    skin_ratio = np.sum(skin) / 14400.0
    
    
    is_green = (g == max_rgb) & (sat > 0.18) & (max_rgb > 30) & (max_rgb < 230) & valid
    is_yellow_green = (g > 50) & (r > 40) & (b < 135) & (g >= r*0.6) & (sat > 0.16) & valid
    green_ratio = (np.sum(is_green) + np.sum(is_yellow_green)*0.7) / 14400.0

    
    if skin_ratio > 0.06:  
        return False, "Human photo not allowed! Upload only tomato leaf 🧑‍🌾❌"
    
    if skin_ratio > 0.03 and green_ratio < 0.18: 
        return False, "Human photo not allowed! Upload only tomato leaf 🧑‍🌾❌"

    white_ratio = np.sum(white_mask) / 14400.0
    if white_ratio > 0.45:
        return False, "Document/Poster! Upload only leaf 📄"

    if green_ratio < 0.12:
        return False, "Not a leaf! Upload only tomato leaf 🍃"

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
        is_leaf, msg = is_tomato_leaf_strict(image)
        if not is_leaf:
            st.markdown(f'<div class="error-box"><div class="error-text">❌ {msg}</div></div>', unsafe_allow_html=True)
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
            st.markdown(f'<div class="result-box"><div class="disease-text">🦠 {pred_name}</div><div class="confidence-text">📊 {confidence:.2f}%</div></div>', unsafe_allow_html=True)
    elif uploaded is None:
        st.info(" Upload tomato leaf image")

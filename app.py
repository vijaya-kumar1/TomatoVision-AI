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
h2 {text-align: center !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;'>🍅 TomatoVision <span style='color:#E53935;'>AI</span></h2>", unsafe_allow_html=True)
st.divider()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")


if not os.path.exists(MODEL_PATH):
    file_id = "1Z-M2gkm79eRnm8HARBIA_c8Jg82Zw4Xg"
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Model downloading... First time 1 min aagum da!"):
        gdown.download(url, MODEL_PATH, quiet=False)

CLASS_NAMES = ["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


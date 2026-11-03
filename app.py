import os
import streamlit as st
st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")
GOOGLE_DRIVE_FILE_ID = "1Z-M2gkm79eRnm8HARBIA_c8Jg82Zw4Xg"
CLASS_NAMES = ["Bacterial_spot","Early_blight","Late_blight","Leaf_Mold","Septoria_leaf_spot","Spider_mites","Target_Spot","Yellow_Leaf_Curl_Virus","Mosaic_virus","healthy"]



@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Downloading model... First time 2 mins VJ!"):
            try:
                
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Model download failed: {e}. Drive link public ah check pannu da!")
                st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


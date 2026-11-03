import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")

MODEL_PATH = "tomato_model_v4.keras"
GOOGLE_DRIVE_FILE_ID = "1Z-M2gkm79eRnmBHARBIA_c83gB2zW4Xg"

def download_model(file_id, dest):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    r = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for k, v in r.cookies.items():
        if k.startswith('download_warning'):
            token = v
            break
    if token:
        r = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Model download panren da VJ... 2 mins wait pannu..."):
            download_model(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
               'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato_Mosaic_virus', 'Tomato_healthy']

model = load_trained_model()
st.title("🍅 TomatoVision AI - Leaf Disease Detector")
uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    if st.button("Disease Kandu Pidi"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.balloons()

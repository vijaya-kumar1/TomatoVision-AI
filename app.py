import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")


MODEL_PATH = "tomato_model_v4.keras"
GOOGLE_DRIVE_FILE_ID = "1-0A2j9k0XvXvXvXvXvXvXvXvXvXvXvX" 

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Model download panren da... First time 2 mins VJ!"):
            try:
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Model download failed: {e}. Drive link public ah check pannu da!")
                st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
               'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
               'Tomato_Mosaic_virus', 'Tomato_healthy']

model = load_trained_model()

st.title("🍅 TomatoVision AI - Leaf Disease Detector")
st.write("Tomato leaf photo va upload pannu da, disease enna nu naan solluren!")

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    if st.button("Disease Kandu Pidi"):
        with st.spinner("Analyzing..."):
            img = image.resize((224, 224)) 
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            st.success(f"**Prediction: {predicted_class}**")
            st.info(f"Confidence: {confidence:.2f}%")

import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")
st.title("🍅 TomatoVision AI")
st.write("Upload a tomato leaf image to predict its class.")
st.warning("⚠️ Please upload ONLY tomato leaf photos. Other images will give wrong results!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model_v4.keras")

CLASS_NAMES = [
 "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
 "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
 "Yellow_Leaf_Curl_Virus", "Mosaic_virus", "healthy",
]

@st.cache_resource
def load_trained_model():
 return tf.keras.models.load_model(MODEL_PATH, compile=False)

def is_tomato_leaf(image):
    
    img_small = image.resize((100, 100))
    arr = np.array(img_small)
    
    white_pixels = np.sum(np.all(arr > 200, axis=2)) / (100*100)
    if white_pixels > 0.6: 
        return False, "This looks like a document/paper, not a leaf!"
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green_dominant = np.sum((g > r+10) & (g > b+10)) / (100*100)
    if green_dominant < 0.15:
        return False, "No green leaf found! Upload only tomato leaf."
    return True, "OK"

uploaded = st.file_uploader("Choose a tomato leaf image", type=["jpg","jpeg","png"])

if uploaded is not None:
 image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
 st.image(image, caption="Uploaded image", width=320)
 
 is_leaf, msg = is_tomato_leaf(image)
 if not is_leaf:
     st.error(f"❌ {msg} Please upload a clear tomato leaf photo!")
     st.stop()

 if st.button("🍅 Predict Disease", type="primary"):
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
      st.warning(f"⚠️ Low confidence ({confidence:.2f}%). This may not be a tomato leaf or image is unclear.")
  
  st.success(f"Prediction: {pred_name}")
  st.metric("Model confidence score", f"{confidence:.2f}%")

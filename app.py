import glob
import os


os.environ["USE_TF"] = "0"

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


st.set_page_config(page_title="TomatoVision AI", page_icon="🍅", layout="wide")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Yellow_Leaf_Curl_Virus",
    "Mosaic_virus",
    "healthy",
]

DISPLAY_NAMES = {
    "Bacterial_spot": "Bacterial Spot",
    "Early_blight": "Early Blight",
    "Late_blight": "Late Blight",
    "Leaf_Mold": "Leaf Mold",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Spider_mites": "Spider Mites",
    "Target_Spot": "Target Spot",
    "Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Mosaic_virus": "Mosaic Virus",
    "healthy": "Healthy Leaf",
}


LEAF_THRESHOLD = 0.5

LEAF_PROMPTS = [
    "a close-up photo of a tomato leaf",
    "a photo of a diseased tomato plant leaf",
    "a photo of a green plant leaf",
    "a photo of a leaf with brown spots",
]
OTHER_PROMPTS = [
    "a photo of a person",
    "a photo of an animal",
    "a photo of a building",
    "a photo of food",
    "a photo of a car",
    "a photo of a room interior",
    "a screenshot or a document",
    "a photo of an object on a table",
    "a photo of a landscape or sky",
    "a photo of a flower",
    "a photo of a fruit",
    "a cartoon or a drawing",
]
ALL_PROMPTS = LEAF_PROMPTS + OTHER_PROMPTS


# ======================================================
# 3. LOAD MODELS
# ======================================================
def find_model():
    """Newest .keras file in the same folder as app.py."""
    matches = glob.glob(os.path.join(BASE_DIR, "*.keras"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="First time only: loading leaf-check model (about 600 MB download)...")
def load_clip():
    """Returns (model, processor) or None if torch / transformers are not installed."""
    try:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        return clip_model, clip_processor
    except Exception:
        return None


# ======================================================
# 4. HELPERS
# ======================================================
def check_leaf(image):
    """Returns (is_leaf, leaf_score). Returns (None, None) if the check is not available."""
    loaded = load_clip()
    if loaded is None:
        return None, None

    import torch

    clip_model, clip_processor = loaded
    inputs = clip_processor(
        text=ALL_PROMPTS, images=image.convert("RGB"), return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image[0]
    probs = torch.softmax(logits, dim=0).numpy()
    leaf_score = float(probs[: len(LEAF_PROMPTS)].sum())
    return leaf_score >= LEAF_THRESHOLD, leaf_score


def prepare_image(image, size):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((size, size))
    # NO /255 here. The model already has preprocessing inside it.
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


# ======================================================
# 5. STYLE
# ======================================================
st.markdown(
    """
<style>
.stApp { background-color: #10131a; color: white; }
h1 { color: #ff5555; }
.result-card {
    background-color: #1c222d;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #343b49;
}
.small-label { color: #b9c6d8; font-size: 16px; }
.result-name { color: #ffffff; font-size: 28px; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🍅 TomatoVision AI")
st.caption("Tomato Leaf Disease Detection | Upload a leaf image to predict its class")
st.divider()

# ======================================================
# 6. LAYOUT
# ======================================================
left_col, right_col = st.columns([1, 1.2], gap="large")

image = None
predict_button = False

with left_col:
    st.subheader("📷 Upload Tomato Leaf")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image")
            predict_button = st.button("🔍 Predict Disease", type="primary")
        except Exception as e:
            st.error(f"Unable to open image: {e}")
            image = None
    else:
        st.info("Upload a tomato leaf image to begin.")

with right_col:
    st.subheader("🧪 Prediction Result")

    if image is not None and predict_button:
        model_path = find_model()

        if model_path is None:
            st.error("Model file not found.")
            st.info(
                "Put your .keras model file (example tomato_model_v4.keras) in the "
                f"same folder as app.py ({BASE_DIR})."
            )
        else:
            try:
                # ---------- Step 1: is it a leaf photo? ----------
                is_leaf, leaf_score = check_leaf(image)

                if is_leaf is False:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="small-label">Result</div>
                            <div class="result-name">Not a leaf image ❌</div>
                            <p>This photo does not look like a tomato leaf
                            (leaf score: {leaf_score * 100:.0f}%).<br>
                            Please upload a clear photo of a tomato leaf.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    if is_leaf is None:
                        st.info(
                            "Leaf check is not active. To turn it on, run: "
                            "pip install torch transformers  (then restart the app)."
                        )

                    # ---------- Step 2: predict the disease ----------
                    with st.spinner("Analyzing leaf image..."):
                        model = load_model(model_path)
                        size = model.input_shape[1] or 224
                        batch = prepare_image(image, size)
                        scores = np.asarray(model.predict(batch, verbose=0)[0]).reshape(-1)

                    if len(scores) != len(CLASS_NAMES):
                        st.error(
                            f"Model gives {len(scores)} outputs but CLASS_NAMES has "
                            f"{len(CLASS_NAMES)} names."
                        )
                    else:
                        if not np.isclose(scores.sum(), 1.0, atol=1e-3):
                            scores = tf.nn.softmax(scores).numpy()

                        top_index = int(np.argmax(scores))
                        predicted = CLASS_NAMES[top_index]
                        confidence = float(scores[top_index]) * 100

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="small-label">Predicted disease / class</div>
                                <div class="result-name">{DISPLAY_NAMES[predicted]}</div>
                                <br>
                                <div class="small-label">Model confidence</div>
                                <h1>{confidence:.2f}%</h1>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(confidence / 100, 0.0), 1.0))

                        if leaf_score is not None:
                            st.caption(f"Leaf check passed ✅ (leaf score {leaf_score * 100:.0f}%)")

                        if confidence < 60:
                            st.warning(
                                "The model is not very sure. Try a clear, close photo "
                                "of one leaf."
                            )

                        st.subheader("🏅 Top 3 Predictions")
                        top3 = np.argsort(scores)[::-1][:3]
                        top3_df = pd.DataFrame(
                            {
                                "Disease": [DISPLAY_NAMES[CLASS_NAMES[i]] for i in top3],
                                "Confidence (%)": [round(float(scores[i]) * 100, 2) for i in top3],
                            }
                        )
                        st.table(top3_df)

                        st.subheader("📊 Confidence for All Classes")
                        chart_df = pd.DataFrame(
                            {
                                "Class": [DISPLAY_NAMES[n] for n in CLASS_NAMES],
                                "Confidence (%)": scores * 100,
                            }
                        ).sort_values("Confidence (%)", ascending=False)
                        st.bar_chart(chart_df.set_index("Class"))

                        st.caption(
                            f"Model file: {os.path.basename(model_path)} | "
                            "Confidence is the model's score, not a guarantee."
                        )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="small-label">Predicted class</div>
                <div class="result-name">Waiting for image...</div>
                <p>Upload a leaf photo on the left, then click Predict Disease.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "TomatoVision AI | AI-based predictions may be incorrect. "
    "Verify important plant-health decisions with an expert."
)

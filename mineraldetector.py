import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("Mineral Classifier")

CLASS_NAMES = ["azurite", "copper", "malachite", "pyrite", "wulfenite"]
IMG_SIZE = (384, 384)

@st.cache_resource
def load_model():
    model_dir = "final_best_model_savedmodel"
    if not tf.io.gfile.exists(model_dir):
        raise FileNotFoundError(
            f"SavedModel directory not found: {model_dir}. "
            "Export and upload it before deploying."
        )

    loaded = tf.saved_model.load(model_dir)
    serving_fn = loaded.signatures.get("serving_default")
    if serving_fn is None:
        raise ValueError("SavedModel has no 'serving_default' signature")
    return serving_fn


def predict_probs(serving_fn, x: np.ndarray):
    outputs = serving_fn(tf.convert_to_tensor(x, dtype=tf.float32))
    first_key = next(iter(outputs))
    probs = outputs[first_key].numpy()[0]
    return probs


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


model = load_model()
uploaded = st.file_uploader("Upload mineral image", type=[
                            "jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = preprocess_image(image)
    probs = predict_probs(model, x)
    pred_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: {CLASS_NAMES[pred_idx]}")
    st.write("Confidence:", f"{probs[pred_idx]*100:.2f}%")

    st.subheader("Class probabilities")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p:.4f}")

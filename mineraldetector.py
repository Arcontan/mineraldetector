import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("Mineral Classifier")

CLASS_NAMES = ["azurite", "copper", "malachite", "pyrite", "wulfenite"]
IMG_SIZE = (384, 384)


@st.cache_resource
def load_model():
    app_dir = Path(__file__).resolve().parent
    keras_path = app_dir / "final_best_model.keras"
    h5_path = app_dir / "final_best_model.h5"

    if keras_path.exists():
        return tf.keras.models.load_model(str(keras_path), compile=False)
    if h5_path.exists():
        return tf.keras.models.load_model(str(h5_path), compile=False)

    raise FileNotFoundError(
        f"No model file found at {keras_path} or {h5_path}")


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
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: {CLASS_NAMES[pred_idx]}")
    st.write("Confidence:", f"{probs[pred_idx]*100:.2f}%")

    st.subheader("Class probabilities")
    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p:.4f}")

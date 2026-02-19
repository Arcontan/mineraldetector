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
    model_path = app_dir / "final_best_model.keras"

    if not model_path.exists():
        files = sorted([p.name for p in app_dir.iterdir()])
        st.error(
            "Model file not found. Expected file at: "
            f"{model_path}\n\n"
            f"App directory: {app_dir}\n"
            f"Files present: {files}"
        )
        st.stop()

    return tf.keras.models.load_model(model_path, compile=False)


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

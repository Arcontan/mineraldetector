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
    return tf.keras.models.load_model("final_best_model.h5", compile=False, safe_mode=False)


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

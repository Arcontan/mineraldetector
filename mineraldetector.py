import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("Mineral Classifier")

CLASS_NAMES = ["azurite", "copper", "malachite", "pyrite", "wulfenite"]
IMG_SIZE = (384, 384)
MODEL_DIR = "final_best_model_savedmodel"

@st.cache_resource
def load_model():
    if not tf.io.gfile.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"SavedModel directory not found: {MODEL_DIR}. "
            "Export and upload it before deploying."
        )
    try:
        loaded = tf.saved_model.load(MODEL_DIR)
    except AttributeError as exc:
        if "add_slot" in str(exc):
            raise RuntimeError(
                "SavedModel contains optimizer slot state that is incompatible with this runtime. "
                "Re-export from training notebook with: "
                "final_best_model.save('final_best_model_savedmodel', save_format='tf', include_optimizer=False)"
            ) from exc
        raise

    serving_fn = loaded.signatures.get("serving_default")
    if serving_fn is None and loaded.signatures:
        serving_fn = next(iter(loaded.signatures.values()))
    if serving_fn is None:
        raise ValueError("SavedModel has no callable signatures")
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


try:
    model = load_model()
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.info(
        "Re-export from notebook using: "
        "final_best_model.save('final_best_model_savedmodel', save_format='tf', include_optimizer=False)"
    )
    st.stop()

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

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
    candidate_paths = [
        app_dir / "final_best_model.h5",
        app_dir / "final_best_model.keras",
    ]

    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    def _load(path: Path):
        try:
            return tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
        except TypeError:
            return tf.keras.models.load_model(str(path), compile=False)
        except Exception as exc:
            msg = str(exc)
            if "bad marshal data" in msg or "Lambda" in msg:
                raise RuntimeError(
                    "Model deserialization failed due to Python-version/Lambda incompatibility. "
                    "This model was saved with Lambda layers; deploy with a compatible Python runtime "
                    "(recommended: Python 3.9 + TensorFlow 2.10.1) or re-export the model without Lambda layers."
                ) from exc
            raise

    errors = []
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            return _load(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    available_files = sorted([p.name for p in app_dir.iterdir()])
    if errors:
        raise RuntimeError(
            "Found model file(s), but failed to load all candidates.\n"
            f"Tried: {[p.name for p in candidate_paths]}\n"
            f"Errors: {errors}\n"
            f"Files in app directory: {available_files}"
        )

    raise FileNotFoundError(
        "No model file found. "
        f"Expected one of: {[p.name for p in candidate_paths]}\n"
        f"Files in app directory: {available_files}"
    )


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

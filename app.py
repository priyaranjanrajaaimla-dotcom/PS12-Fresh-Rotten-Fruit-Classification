import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_PATH = "PS12_fruit_freshness_model.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["freshapples", "freshbanana", "freshoranges",
               "rottenapples", "rottenbanana", "rottenoranges"]

FRUIT_NAMES = {
    "freshapples": "Apple", "freshbanana": "Banana", "freshoranges": "Orange",
    "rottenapples": "Apple", "rottenbanana": "Banana", "rottenoranges": "Orange"
}
CONDITION_NAMES = {
    "freshapples": "Fresh", "freshbanana": "Fresh", "freshoranges": "Fresh",
    "rottenapples": "Rotten", "rottenbanana": "Rotten", "rottenoranges": "Rotten"
}

st.set_page_config(page_title="Fresh vs Rotten Fruit Classifier", page_icon="🍎")
st.title("🍎 Fresh vs Rotten Fruit Classifier")
st.write("Upload an apple, banana, or orange image. The trained model predicts its fruit type and condition.")
st.info("PS-12 | MobileNetV2 transfer learning | Input size: 160 × 160")

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load '{MODEL_PATH}'. Make sure the model file is in the same folder as app.py.")
    st.code(str(exc))
    st.stop()

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image you uploaded", use_container_width=True)
        resized = image.resize(IMG_SIZE)
        image_batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

        if st.button("Classify fruit", type="primary", use_container_width=True):
            with st.spinner("Analysing image..."):
                scores = model.predict(image_batch, verbose=0)[0]

            index = int(np.argmax(scores))
            class_name = CLASS_NAMES[index]
            fruit = FRUIT_NAMES[class_name]
            condition = CONDITION_NAMES[class_name]
            confidence = float(scores[index]) * 100

            st.subheader("Prediction result")
            col1, col2 = st.columns(2)
            col1.metric("Fruit type", fruit)
            col2.metric("Condition", condition)
            st.metric("Model confidence", f"{confidence:.2f}%")

            if condition == "Fresh":
                st.success(f"Prediction: Fresh {fruit.lower()}")
            else:
                st.warning(f"Prediction: Rotten {fruit.lower()}")

            st.caption("Confidence is the model's score, not a guarantee of correctness. This demo does not determine food safety.")
            with st.expander("View scores for all six classes"):
                for name, score in sorted(zip(CLASS_NAMES, scores), key=lambda x: x[1], reverse=True):
                    label = f"{FRUIT_NAMES[name]} — {CONDITION_NAMES[name]}"
                    st.write(f"{label}: {float(score) * 100:.2f}%")
                    st.progress(float(score))
    except Exception as exc:
        st.error("Could not process this image. Try a JPG or PNG file.")
        st.code(str(exc))
else:
    st.caption("Tip: choose a clear image with one fruit visible for your classroom demonstration.")

st.divider()
st.caption("PS-12 Fresh vs Rotten Fruit Sorting | Educational demonstration")

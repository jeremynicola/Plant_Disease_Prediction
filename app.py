import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Plant Leaf Disease Prediction", layout="centered")
st.title("🌿 Plant Leaf Disease Prediction")
st.write("Upload a leaf image and I will predict the disease (if any).")

# ----------------- Load Model -----------------
@st.cache_data(show_spinner=True)
def load_model():
    model = tf.keras.models.load_model("leaf_disease_model.keras")
    with open("label_encoder.pkl", "rb") as f:
        class_indices = pickle.load(f)
    # Reverse mapping for class names
    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names

model, CLASS_NAMES = load_model()

# ----------------- Prediction Function -----------------
def predict(image, model):
    image = image.resize((160,160))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    probs = model.predict(img_array)[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100
    status = "Healthy" if "healthy" in pred_class.lower() else "Diseased"
    return pred_class, status, confidence, probs

# ----------------- Upload Section -----------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Predicting..."):
        pred_class, status, confidence, probs = predict(image, model)
    
    st.success("✅ Prediction")
    st.write(f"### Class: {pred_class}")
    st.write(f"**Status:** {status}  |  **Confidence:** {confidence:.2f}%")

    # ----------------- Probability Bar Chart -----------------
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(list(CLASS_NAMES.values()), probs, color='green')
    ax.set_xlabel("Probability")
    ax.set_xlim(0,1)
    
    # Highlight predicted class in red
    pred_idx = np.argmax(probs)
    ax.get_children()[pred_idx].set_color('red')
    
    st.pyplot(fig)

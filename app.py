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

# ----------------- Remedies Dictionary -----------------
REMEDIES = {
    "Apple_Apple_scab": [
        "💊 Apply fungicides like **captan** or **mancozeb**.",
        "✂️ Prune infected leaves and branches.",
        "💧 Avoid overhead watering."
    ],
    "Apple_Black_rot": [
        "🔥 Remove mummified fruits and infected branches.",
        "💊 Use fungicides (thiophanate-methyl, captan).",
        "🧹 Practice crop rotation and sanitation."
    ],
    "Apple_Cedar_apple_rust": [
        "🌱 Plant resistant apple varieties.",
        "🌳 Remove nearby juniper trees (alternate host).",
        "💊 Apply preventive fungicides during spring."
    ],
    "Apple_healthy": [
        "✅ No disease detected!",
        "💧 Maintain regular watering and nutrition.",
        "🔍 Monitor regularly for early signs of disease."
    ],

    "Blueberry_healthy": [
        "✅ No disease detected!",
        "☀️ Ensure good sunlight and soil drainage.",
        "🌱 Fertilize as recommended."
    ],

    "Cherry_Powdery_mildew": [
        "💊 Apply sulfur-based fungicides.",
        "🍃 Ensure good air circulation.",
        "✂️ Prune infected leaves."
    ],
    "Cherry_healthy": [
        "✅ No disease detected!",
        "✂️ Maintain pruning and irrigation.",
        "🍂 Use mulch to retain soil moisture."
    ],

    "Corn_Cercospora_leaf_spot_Gray_leaf_spot": [
        "🔄 Rotate crops to break fungal cycle.",
        "🌽 Use resistant hybrids.",
        "💊 Apply fungicides if needed."
    ],
    "Corn_Common_rust": [
        "🌽 Plant resistant corn varieties.",
        "💊 Apply fungicides (triazoles, strobilurins).",
        "🍃 Improve airflow by avoiding dense planting."
    ],
    "Corn_Northern_Leaf_Blight": [
        "🔥 Remove infected debris after harvest.",
        "💊 Apply fungicides early.",
        "🌽 Use resistant hybrids."
    ],
    "Corn_healthy": [
        "✅ No disease detected!",
        "💧 Maintain soil fertility and irrigation.",
        "🔍 Regular scouting for disease signs."
    ],

    "Grape_Black_rot": [
        "✂️ Remove infected leaves and fruits.",
        "💊 Apply fungicides (myclobutanil, mancozeb).",
        "🍃 Prune vines to improve airflow."
    ],
    "Grape_Esca_Black_Measles": [
        "✂️ Prune and remove infected wood.",
        "💧 Avoid vine stress (water properly).",
        "⚠️ No chemical cure – focus on prevention."
    ],
    "Grape_Leaf_blight_Isariopsis_Leaf_Spot": [
        "💊 Apply copper-based fungicides.",
        "✂️ Remove infected leaves.",
        "🧹 Improve field sanitation."
    ],
    "Grape_healthy": [
        "✅ No disease detected!",
        "🌱 Maintain balanced fertilization.",
        "✂️ Ensure proper pruning."
    ],

    "Orange_Haunglongbing_Citrus_greening": [
        "🔥 Remove infected trees immediately.",
        "🐞 Control psyllid insect vector.",
        "🌱 Use disease-free planting material."
    ],

    "Peach_Bacterial_spot": [
        "💊 Use copper-based bactericides.",
        "💧 Avoid overhead irrigation.",
        "✂️ Remove infected leaves."
    ],
    "Peach_healthy": [
        "✅ No disease detected!",
        "🧹 Maintain orchard hygiene.",
        "🌱 Use disease-free planting material."
    ],

    "Pepper_Bacterial_spot": [
        "🌱 Use disease-free seeds.",
        "💊 Apply copper fungicides.",
        "⛔ Avoid working in fields when wet."
    ],
    "Pepper_healthy": [
        "✅ No disease detected!",
        "🌱 Provide balanced nutrition.",
        "🔍 Monitor regularly for disease."
    ],

    "Potato_Early_blight": [
        "🌱 Use resistant potato varieties.",
        "💊 Apply fungicides (chlorothalonil, mancozeb).",
        "🔄 Rotate crops and avoid monoculture."
    ],
    "Potato_Late_blight": [
        "🔥 Destroy infected plants immediately.",
        "💊 Use fungicides (metalaxyl, mancozeb).",
        "💧 Avoid waterlogged conditions."
    ],
    "Potato_healthy": [
        "✅ No disease detected!",
        "🌱 Maintain soil health.",
        "🔍 Scout for blight symptoms after rain."
    ],

    "Raspberry_healthy": [
        "✅ No disease detected!",
        "💧 Ensure good drainage.",
        "🍃 Provide trellis support for airflow."
    ],

    "Soybean_healthy": [
        "✅ No disease detected!",
        "🔄 Rotate crops with maize or wheat.",
        "🌱 Ensure balanced fertilization."
    ],

    "Squash_Powdery_mildew": [
        "💊 Spray sulfur or neem oil early.",
        "🌱 Avoid excess nitrogen fertilizer.",
        "🌿 Plant resistant squash varieties."
    ],

    "Strawberry_Leaf_scorch": [
        "✂️ Remove and destroy infected leaves.",
        "🍃 Ensure good air circulation.",
        "💊 Apply fungicides like captan."
    ],
    "Strawberry_healthy": [
        "✅ No disease detected!",
        "🍂 Mulch plants to prevent soil splash.",
        "💧 Maintain good irrigation practices."
    ],

    "Tomato_Bacterial_spot": [
        "💊 Use copper fungicides regularly.",
        "💧 Avoid overhead irrigation.",
        "🌱 Plant resistant varieties."
    ],
    "Tomato_Early_blight": [
        "💊 Apply fungicides (chlorothalonil, mancozeb).",
        "✂️ Prune infected leaves.",
        "🔄 Rotate crops yearly."
    ],
    "Tomato_Late_blight": [
        "🔥 Destroy infected plants immediately.",
        "💊 Apply fungicides (metalaxyl, chlorothalonil).",
        "💧 Avoid waterlogged fields."
    ],
    "Tomato_Leaf_Mold": [
        "🍃 Ensure greenhouse ventilation.",
        "💊 Apply fungicides (chlorothalonil, mancozeb).",
        "🌱 Avoid overcrowding of plants."
    ],
    "Tomato_Septoria_leaf_spot": [
        "✂️ Prune affected leaves.",
        "💧 Use drip irrigation.",
        "💊 Apply preventive fungicides."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "🌿 Spray neem oil or insecticidal soap.",
        "🐞 Encourage natural predators (ladybugs).",
        "⛔ Avoid excessive pesticide use."
    ],
    "Tomato_Target_Spot": [
        "💊 Apply fungicides (chlorothalonil, mancozeb).",
        "🍃 Improve air circulation.",
        "🔄 Rotate crops annually."
    ],
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": [
        "🔥 Remove infected plants immediately.",
        "🐞 Control whiteflies (vector).",
        "🌱 Plant resistant varieties."
    ],
    "Tomato_Tomato_mosaic_virus": [
        "🔥 Remove and destroy infected plants.",
        "🧼 Disinfect tools and hands before handling plants.",
        "🌱 Use resistant tomato cultivars."
    ],
    "Tomato_healthy": [
        "✅ No disease detected!",
        "🌱 Maintain balanced nutrition and watering.",
        "🔍 Scout for pests and diseases regularly."
    ]
}

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
    
    with st.spinner("🔎 Analyzing..."):
        pred_class, status, confidence, probs = predict(image, model)
    
    st.success("✅ Prediction Complete")
    st.write(f"### 🏷️ Class: **{pred_class}**")
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

    # ----------------- Remedies Section -----------------
    st.subheader("🌱 Suggested Remedies")

    DEFAULT_REMEDIES = [
        "🧹 Maintain good field hygiene (remove diseased leaves/plants).",
        "💧 Avoid overhead watering to reduce leaf wetness.",
        "🔄 Rotate crops to prevent soil-borne diseases.",
        "🌱 Use resistant varieties if available.",
        "👨‍🌾 Consult local agricultural experts for region-specific advice."
    ]

    remedies = REMEDIES.get(pred_class, DEFAULT_REMEDIES)

    with st.expander("📋 View Remedies"):
        for r in remedies:
            st.markdown(f"- {r}")

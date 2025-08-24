# Leaf Disease Prediction App 🌿

This is a Streamlit app that predicts plant leaf diseases using a pre-trained Keras model.

## Features

- Upload a leaf image (jpg/png)
- Predict disease or healthy status
- Display confidence percentage
- Show horizontal bar chart of all class probabilities

## Files

- `app.py` → Streamlit application
- `leaf_disease_model.keras` → Trained Keras model
- `label_encoder.pkl` → Class label mapping

## Dataset

- Original dataset: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Dataset is **not included** in this repository (large ~600MB)
- To train your own model, download dataset and follow training notebook instructions

## Deployment on Streamlit

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click **New app**
3. Connect your GitHub account
4. Select repository: `Leaf_Disease_Prediction`
5. Choose branch (e.g., `main`) and main file: `app.py`
6. Click **Deploy**
7. App will be live in a few minutes

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- Pillow
- Matplotlib
- Numpy

```bash
pip install streamlit tensorflow pillow matplotlib numpy

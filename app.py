import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Breast Cancer Prediction App")
st.write("Enter all feature values exactly as required by model:")

# 30 feature names
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

inputs = []

# UI inputs for all 30 features
for col in feature_names:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    x = np.array([inputs])  # MUST be (1,30)
    pred = model.predict(x)[0]

    if pred == 1:
        st.error("⚠️ High chance of malignant tumor")
    else:
        st.success("✔ Likely benign tumor")

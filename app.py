# import streamlit as st
# import numpy as np
# import pickle

# # Load model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# st.title("Breast Cancer Prediction App")
# st.write("Enter all feature values exactly as required by model:")

# # 30 feature names
# feature_names = [
#     "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
#     "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
#     "radius error", "texture error", "perimeter error", "area error", "smoothness error",
#     "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
#     "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
#     "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
# ]

# inputs = []

# # UI inputs for all 30 features
# for col in feature_names:
#     val = st.number_input(col, value=0.0)
#     inputs.append(val)

# if st.button("Predict"):
#     x = np.array([inputs])  # MUST be (1,30)
#     pred = model.predict(x)[0]

#     if pred == 1:
#         st.error("⚠️ High chance of malignant tumor")
#     else:
#         st.success("✔ Likely benign tumor")
import streamlit as st
import pickle
import numpy as np

# Load model
with open("model2.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
features = data["features"]

st.title("Breast Cancer Prediction")

# Create input fields
user_input = []
for feature in features:
    val = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(val)

# Prediction
if st.button("Predict"):
    arr = np.array(user_input).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Cancer Detected")
    else:
        st.success("✔️ No Cancer Detected")

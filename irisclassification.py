import streamlit as st
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load("iris_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Title
st.title("🌼 Iris Flower Predictor")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Iris Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = le.inverse_transform(prediction)[0]
    st.success(f"Predicted Iris species: **{species}**")




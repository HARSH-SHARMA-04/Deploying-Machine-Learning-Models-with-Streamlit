
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load('iris_model.pkl')

# App title
st.title("🌸 Iris Flower Classifier")
st.write("This app predicts the **Iris flower** species based on your input!")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Output
classes = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction")
st.write(f"🌼 Predicted class: **{classes[prediction]}**")

# Probability bar chart
st.subheader("Prediction Probability")
fig, ax = plt.subplots()
bars = ax.bar(classes, proba, color=['#FFDDC1', '#FFABAB', '#FFC3A0'])
ax.set_ylim(0, 1)
st.pyplot(fig)

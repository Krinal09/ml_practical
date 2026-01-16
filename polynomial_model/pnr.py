import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("poly_model.pkl", "rb"))
poly = pickle.load(open("poly_transformer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Auto MPG Prediction (Numeric Polynomial Model)")

cylinders = st.slider("Cylinders", 3, 8, 4)
displacement = st.slider("Displacement", 50, 500, 200)
horsepower = st.slider("Horsepower", 40, 250, 100)
weight = st.slider("Weight", 1500, 5000, 3000)
acceleration = st.slider("Acceleration", 5.0, 25.0, 15.0)
model_year = st.slider("Model Year", 70, 82, 76)

input_df = pd.DataFrame([{
    "cylinders": cylinders,
    "displacement": displacement,
    "horsepower": horsepower,
    "weight": weight,
    "acceleration": acceleration,
    "model year": model_year
}])

input_scaled = scaler.transform(input_df)
input_poly = poly.transform(input_scaled)

prediction = model.predict(input_poly)

st.subheader("Predicted MPG")
st.write(f"{prediction[0]:.2f}")
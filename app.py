
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# Load trained model and scaler
model = tf.keras.models.load_model("flight_delay_classifier.keras")
scaler = joblib.load("flight_classifier_scaler.pkl")

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️"
)

st.title("✈️ Flight Delay Prediction System")
st.write("Enter flight details to predict whether the flight may be delayed.")

# Input fields
distance = st.number_input(
    "Distance (km)",
    min_value=1.0,
    value=500.0
)

air_time = st.number_input(
    "Air Time (minutes)",
    min_value=1.0,
    value=90.0
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame(
        [[distance, air_time]],
        columns=["Distance", "AirTime"]
    )

    scaled_data = scaler.transform(input_data)

    probability = model.predict(
        scaled_data,
        verbose=0
    )[0][0]

    threshold = 0.45

    if probability >= threshold:
        st.error("⚠️ Flight is predicted to be DELAYED")
    else:
        st.success("✅ Flight is predicted to be ON TIME")

    st.write(f"Model score: {probability:.2%}")

st.caption(
    "Note: Predictions are based only on Distance and Air Time. "
    "Actual flight delays may depend on other factors."
)

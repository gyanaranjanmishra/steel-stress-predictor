import streamlit as st
import numpy as np
import pickle
import tensorflow as tf


# ----------------------------
# Load Models & Scalers
# ----------------------------
@st.cache_resource
def load_models():

    model_13 = tf.keras.models.load_model("models/model13si.keras")
    model_32 = tf.keras.models.load_model("models/model32si.keras")

    with open("scalers/scaler13si.pkl", "rb") as f:
        scaler_13 = pickle.load(f)

    with open("scalers/scaler32si.pkl", "rb") as f:
        scaler_32 = pickle.load(f)

    return model_13, model_32, scaler_13, scaler_32


model13, model32, scaler13, scaler32 = load_models()


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="High Temperature Flow Stress Predictor")

st.title("Electrical Steel Flow Stress Prediction")

st.write("Predict stress using strain, strain rate, and temperature and Si content")

# Select steel type
steel_type = st.selectbox(
    "Select Silicon Content",
    ["1.3 wt% Si", "3.2 wt% Si"]
)

# Inputs
strain = st.number_input("Strain", value=0.0)
strain_rate = st.number_input("Strain Rate (1/s)", value=0.0)
temperature = st.number_input("Temperature (°C)", value=25.0)


# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Stress"):

    X = np.array([[strain, strain_rate, temperature]])

    if steel_type == "1.3 wt% Si":

        X_scaled = scaler13.transform(X)
        pred = model13.predict(X_scaled)[0][0]

    else:

        X_scaled = scaler32.transform(X)
        pred = model32.predict(X_scaled)[0][0]

    st.success("Prediction Complete!")
    st.write(f"### Predicted Stress: **{pred:.2f} MPa**")
import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/flood_prediction_model.pkl")

st.sidebar.title("Flood Prediction System")

st.sidebar.markdown("---")


st.sidebar.info(
    """
    **Machine Learning Models**
    - Random Forest
    - Decision Tree
    - Logistic Regression

    **Dataset Size**
    - 2,192 Records

    **Best Model**
    - Random Forest
    """
)

# Page settings
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="centered"
)

# Title
st.title("🌊 Flood Prediction System")
st.markdown(
    """
    Welcome to the **Flood Prediction System**.

    Enter the weather conditions below to estimate the flood risk using a trained **Random Forest Machine Learning model**.
    """
)

st.write("Predict flood risk using weather conditions.")

# Input fields
temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=25.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

# Prediction
if st.button("Predict"):

    sample = pd.DataFrame({
        "Temperature_C": [temperature],
        "Rainfall_mm": [rainfall],
        "Humidity_%": [humidity]
    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Flood Risk Level: HIGH")
    else:
        st.success("✅ Flood Risk Level: LOW")

    st.metric(
        label="Flood Probability",
        value=f"{probability * 100:.2f}%"
    )

    st.markdown("---")

    st.caption(
        "This prediction is generated using a Random Forest Machine Learning model trained on historical weather data."
    )
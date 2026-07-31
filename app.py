import streamlit as st
import joblib
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="centered"
)

# Load trained model
model = joblib.load("models/flood_prediction_model.pkl")
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
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

st.subheader("🌦️ Enter Weather Conditions")
st.write("Provide the current weather values to estimate the flood risk.")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0
    )

with col2:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=50.0
    )

with col3:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

st.markdown("---")
st.subheader("🔍 Flood Risk Prediction")


# Prediction
if st.button("Predict"):

    sample = pd.DataFrame({
        "Temperature_C": [temperature],
        "Rainfall_mm": [rainfall],
        "Humidity_%": [humidity]
    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.session_state.history.append({
        "Temperature": temperature,
        "Rainfall": rainfall,
        "Humidity": humidity,
        "Risk": "HIGH" if prediction == 1 else "LOW",
        "Probability": f"{probability * 100:.2f}%"
    })

    st.divider()
    st.subheader("Prediction Result")
    st.caption("The result below is based on the weather conditions you entered.")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("✅ LOW RISK")

    with result_col2:
        st.metric(
            label="Flood Probability",
            value=f"{probability * 100:.2f}%"
        )

    if prediction == 1:
        st.warning(
            """
            **Safety Recommendation**

            • Stay alert and monitor weather updates.

            • Avoid low-lying and flood-prone areas.

            • Keep important documents and emergency supplies ready.
            """
        )
    else:
        st.info(
            """
            **Safety Recommendation**

            • Current weather conditions indicate a low flood risk.

            • Continue monitoring rainfall and weather conditions.
            """
        )

    st.markdown("---")

    st.caption(
        "This prediction is generated using a Random Forest Machine Learning model trained on historical weather data."
    )

if st.session_state.history:
    st.markdown("---")
    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


st.markdown("---")
st.subheader("📊 Model Performance")

performance_df = pd.read_csv("results/model_performance.csv")

best_model_row = performance_df.loc[
    performance_df["Accuracy"].idxmax()
]

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "Best Model",
        best_model_row["Model"]
    )

with metric_col2:
    st.metric(
        "Best Accuracy",
        f"{best_model_row['Accuracy'] * 100:.2f}%"
    )

with metric_col3:
    st.metric(
        "Best F1-Score",
        f"{best_model_row['F1-Score'] * 100:.2f}%"
    )

st.dataframe(
    performance_df,
    use_container_width=True
)
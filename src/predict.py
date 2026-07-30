import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/flood_prediction_model.pkl")

print("=" * 50)
print("Flood Prediction System")
print("=" * 50)

# User Input
temperature = float(input("Enter Temperature (°C): "))
rainfall = float(input("Enter Rainfall (mm): "))
humidity = float(input("Enter Humidity (%): "))

# Create DataFrame
data = pd.DataFrame({
    "Temperature_C": [temperature],
    "Rainfall_mm": [rainfall],
    "Humidity_%": [humidity]
})

# Prediction
prediction = model.predict(data)

print("\n" + "=" * 50)

if prediction[0] == 1:
    print("Flood Risk Level : HIGH")
else:
    print("Flood Risk Level : LOW")
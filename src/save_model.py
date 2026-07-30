import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/processed/flood_dataset.csv")

# Features
X = df[["Temperature_C", "Rainfall_mm", "Humidity_%"]]

# Target
y = df["Flood_Risk"]

# Train Random Forest
model = RandomForestClassifier(random_state=42)

model.fit(X, y)

# Save model
joblib.dump(model, "models/flood_prediction_model.pkl")

print("=" * 50)
print("Model saved successfully!")
print("Location : models/flood_prediction_model.pkl")
print("=" * 50)
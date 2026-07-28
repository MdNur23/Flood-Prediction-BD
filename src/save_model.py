import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/processed/flood_dataset.csv")

# Features
X = df[["Temperature_C", "Rainfall_mm", "Humidity_%"]]

# Target
y = df["Flood_Risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/flood_model.pkl")

print("=" * 50)
print("Model Saved Successfully!")
print("=" * 50)
print("Location: models/flood_model.pkl")
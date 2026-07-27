import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Prediction
y_pred = model.predict(X_test)

# Results
print("= - train_model.py:32" * 50)
print("Decision Tree Results - train_model.py:33")
print("= - train_model.py:34" * 50)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f} - train_model.py:36")

print("\nClassification Report: - train_model.py:38")
print(classification_report(y_test, y_pred))
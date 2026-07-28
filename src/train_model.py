import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

print("=" * 50)
print("Flood Prediction Model Comparison")
print("=" * 50)

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name:<22} Accuracy: {accuracy:.4f}")

print("=" * 50)

best_model = max(results, key=results.get)

print(f"Best Model : {best_model}")
print(f"Accuracy   : {results[best_model]:.4f}")
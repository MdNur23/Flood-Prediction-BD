import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

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
    cm = confusion_matrix(y_test, y_pred)

    results[name] = accuracy

    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Flood", "Flood"],
        yticklabels=["No Flood", "Flood"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")

    plt.savefig(
        f"data/processed/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.close()

for name, model in models.items():
    ...
    plt.close()


dt_model = models["Decision Tree"]

feature_names = X.columns
importances = dt_model.feature_importances_

plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.savefig("data/processed/feature_importance.png")
plt.close()



print("=" * 50)

best_model = max(results, key=results.get)

print(f"Best Model : {best_model}")
print(f"Accuracy   : {results[best_model]:.4f}")
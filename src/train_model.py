import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

# Load dataset
df = pd.read_csv("data/processed/flood_dataset.csv")

print("\nDataset Shape:", df.shape)
print(df["Flood_Risk"].value_counts())

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
performance = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = accuracy

    performance.append({
    "Model": name,
    "Accuracy": accuracy,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
})

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

print("\nCross Validation Results")
print("-" * 30)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)

    print(f"{name}")
    print(f"Scores : {scores}")
    print(f"Mean Accuracy : {scores.mean():.4f}")
    print(f"Std Dev : {scores.std():.4f}\n")



print("=" * 50)

best_model = max(results, key=results.get)

print(f"Best Model : {best_model}")
print(f"Accuracy   : {results[best_model]:.4f}")

# Save model performance to CSV
performance_df = pd.DataFrame(performance)

performance_df.to_csv(
    "results/model_performance.csv",
    index=False
)

print("\nModel performance saved to results/model_performance.csv")

# ROC Curve for Random Forest

rf_model = models["Random Forest"]

y_prob = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

print(f"\nROC AUC Score : {roc_auc:.4f}")
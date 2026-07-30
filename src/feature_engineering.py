import os
import pandas as pd
import numpy as np

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/bangladesh_weather.csv")

# Flood Risk Label (Research-Oriented)

score = (
    (df["Rainfall_mm"] * 0.6)
    + (df["Humidity_%"] * 0.3)
    - (df["Temperature_C"] * 0.2)
)

# Add random noise to make prediction more realistic
noise = np.random.normal(0, 5, len(df))

score = score + noise

df["Flood_Risk"] = (score >= 40).astype(int)

# Save new dataset
output = "data/processed/flood_dataset.csv"
df.to_csv(output, index=False)

print("=" * 50)
print("Feature Engineering Completed")
print("=" * 50)

print(df.head())

print("\nFlood Risk Distribution:")
print(df["Flood_Risk"].value_counts())

print(f"\nDataset saved to: {output}")
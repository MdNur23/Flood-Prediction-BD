import os
import pandas as pd

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/bangladesh_weather.csv")

# Flood Risk Label
# Rule:
# Rainfall >= 50 mm/day -> Flood Risk = 1
# Otherwise -> Flood Risk = 0

df["Flood_Risk"] = (df["Rainfall_mm"] >= 50).astype(int)

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
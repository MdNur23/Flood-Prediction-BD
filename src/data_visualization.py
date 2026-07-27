import os
import pandas as pd
import matplotlib.pyplot as plt

# Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/bangladesh_weather.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

# -------- Temperature --------
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Temperature_C"])
plt.title("Bangladesh Temperature (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.savefig("data/processed/temperature_plot.png")
plt.close()

# -------- Rainfall --------
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Rainfall_mm"])
plt.title("Bangladesh Rainfall (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.savefig("data/processed/rainfall_plot.png")
plt.close()

# -------- Humidity --------
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Humidity_%"])
plt.title("Bangladesh Humidity (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.grid(True)
plt.savefig("data/processed/humidity_plot.png")
plt.close()

print("All graphs saved successfully! - data_visualization.py:44")
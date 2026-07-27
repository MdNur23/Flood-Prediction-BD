import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/bangladesh_weather.csv")

print("= - data_analysis.py:6" * 50)
print("Bangladesh Weather Dataset - data_analysis.py:7")
print("= - data_analysis.py:8" * 50)

print("\nFirst 5 Rows: - data_analysis.py:10")
print(df.head())

print("\nDataset Shape: - data_analysis.py:13")
print(df.shape)

print("\nMissing Values: - data_analysis.py:16")
print(df.isnull().sum())

print("\nSummary Statistics: - data_analysis.py:19")
print(df.describe())
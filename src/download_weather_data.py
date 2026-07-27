import requests
import pandas as pd

# Bangladesh (Dhaka) coordinates
LATITUDE = 23.8103
LONGITUDE = 90.4125

# NASA POWER API URL
url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters=T2M,PRECTOTCORR,RH2M"
    f"&community=AG"
    f"&longitude={LONGITUDE}"
    f"&latitude={LATITUDE}"
    f"&start=20200101"
    f"&end=20251231"
    f"&format=JSON"
)

print("Downloading weather data from NASA POWER... - download_weather_data.py:20")

response = requests.get(url, timeout=30)

if response.status_code == 200:
    data = response.json()

    weather = data["properties"]["parameter"]

    df = pd.DataFrame({
        "Date": weather["T2M"].keys(),
        "Temperature_C": weather["T2M"].values(),
        "Rainfall_mm": weather["PRECTOTCORR"].values(),
        "Humidity_%": weather["RH2M"].values()
    })

    output_file = "data/raw/bangladesh_weather.csv"
    df.to_csv(output_file, index=False)

    print(f"\nDataset saved successfully: {output_file} - download_weather_data.py:39")
    print(df.head())

else:
    print("Failed to download data. - download_weather_data.py:43")
    print(response.status_code)
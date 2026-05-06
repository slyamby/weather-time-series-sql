import requests
import pandas as pd
from pathlib import Path

# -----------------------
# Configuration
# -----------------------
OUTPUT_PATH = Path("data/raw/accra_weather.csv")

LATITUDE = 5.6037
LONGITUDE = -0.1870
CITY = "Accra"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_data():
    """
    Fetch daily historical weather data from Open-Meteo
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "Africa/Accra"
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    
    daily_data = data["daily"]

    df = pd.DataFrame(daily_data)

    df["city"] = CITY

    # Reoder columns
    df = df[
        [
            "time",
            "city",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max"
        ]
    ]

    df = df.rename(columns={"time": "date"})

    return df


def save_data(df: pd.DataFrame):
    """
    Save collected weather data to CSV
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Weather data saved to: {OUTPUT_PATH}")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    weather_df = fetch_weather_data()

    save_data(weather_df)
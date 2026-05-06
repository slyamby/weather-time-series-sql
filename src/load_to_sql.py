import pandas as pd
from pathlib import Path
from db import get_connection

CSV_PATH = Path("data/raw/accra_weather.csv")
TABLE_NAME = "weather_daily"


def load_csv_to_sql():
    """
    Load weather CSV data into sqlite table
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)

    conn = get_connection()

    df.to_sql(
        TABLE_NAME,
        conn, 
        if_exists="replace",
        index=False
    )

    conn.close()

    print(f"Loaded {len(df)} rows into table: {TABLE_NAME}")


if __name__ == "__main__":
    load_csv_to_sql()
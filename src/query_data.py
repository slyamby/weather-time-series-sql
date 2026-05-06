import pandas as pd
from db import get_connection

TABLE_NAME = "weather_daily"


def fetch_all_data():
    """
    Fetch full dataset from SQL
    """
    conn = get_connection()

    query = f"""
    SELECT * 
    FROM {TABLE_NAME}
    ORDER BY date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df


def fetch_monthly_summary():
    """
    SQL aggregation for monthly trends
    """
    conn = get_connection()

    query = f"""
    SELECT
        strftime('%Y-%m', date) AS month,
        AVG(temperature_2m_mean) AS avg_temp,
        SUM(precipitation_sum) AS total_rain
    FROM {TABLE_NAME}
    GROUP BY month
    ORDER BY month
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


if __name__ == "__main__":
    df = fetch_all_data()
    print("\nFull dataset")
    print(df.head())

    monthly_df = fetch_monthly_summary()
    print("\nMonthly summary")
    print(monthly_df.head())


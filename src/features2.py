import numpy as np
from query_data import fetch_all_data


def create_time_features(df):
    """
    Create time-based features for forecasting
    """

    df = df.copy()

    # ----------------------------
    # Basic Time Features
    # ----------------------------
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year


    # ---------------------------
    # Cyclic Features
    # ---------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # -----------------------------
    # Yearly cyclic features
    # -----------------------------
    df["day_of_year"] = df.index.dayofyear

    df["year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # -----------------------------
    # Lag features
    # -----------------------------
    df["lag_1"] = df["temperature_2m_mean"].shift(1)
    df["lag_7"] = df["temperature_2m_mean"].shift(7)
    df["lag_30"] = df["temperature_2m_mean"].shift(30)

    # -----------------------------
    # Rolling features
    # -----------------------------
    df["rolling_7"] = df["temperature_2m_mean"].rolling(7).mean()
    df["rolling_30"] = df["temperature_2m_mean"].rolling(30).mean()

    # Drop NA caused by lags
    df = df.dropna()

    return df


if __name__ == "__main__":
    df = fetch_all_data()

    df_features = create_time_features(df)

    print("\nFeature dataset:")
    print(df_features.head())
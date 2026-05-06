import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from query_data import fetch_all_data
from features2 import create_time_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


def recursive_forecast(steps=30):
    """
    Recursive multi-step forecast using a RandomForest lag-based model.

    This predicts one day ahead, adds that prediction back into the history,
    recalculates time/lag/rolling features, then predicts the next day.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. Load and prepare data
    # -----------------------------
    df = fetch_all_data()
    df = create_time_features(df)

    target_col = "temperature_2m_mean"

    y = df[target_col]
    X = df.drop(columns=[target_col, "city"])

    feature_cols = X.columns.tolist()

    # -----------------------------
    # 2. Train model on all available data
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Keep full history for recursive feature updates
    history = df.copy()
    predictions = []

    # -----------------------------
    # 3. Recursive forecasting loop
    # -----------------------------
    for _ in range(steps):

        last_row = history.iloc[-1:].copy()

        X_last = last_row[feature_cols]

        next_pred = model.predict(X_last)[0]
        predictions.append(next_pred)

        next_date = history.index[-1] + pd.Timedelta(days=1)

        new_row = last_row.copy()
        new_row.index = [next_date]

        # Update target
        new_row[target_col] = next_pred

        # -----------------------------
        # Update basic time features
        # -----------------------------
        new_row["day_of_week"] = next_date.dayofweek
        new_row["month"] = next_date.month
        new_row["year"] = next_date.year
        new_row["day_of_year"] = next_date.dayofyear

        # -----------------------------
        # Update cyclic features
        # -----------------------------
        new_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
        new_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)

        new_row["day_sin"] = np.sin(2 * np.pi * next_date.dayofweek / 7)
        new_row["day_cos"] = np.cos(2 * np.pi * next_date.dayofweek / 7)

        new_row["year_sin"] = np.sin(2 * np.pi * next_date.dayofyear / 365)
        new_row["year_cos"] = np.cos(2 * np.pi * next_date.dayofyear / 365)

        # -----------------------------
        # Update lag features
        # -----------------------------
        new_row["lag_1"] = history[target_col].iloc[-1]
        new_row["lag_7"] = history[target_col].iloc[-7]
        new_row["lag_30"] = history[target_col].iloc[-30]

        # -----------------------------
        # Update rolling features
        # -----------------------------
        new_row["rolling_mean_7"] = history[target_col].iloc[-7:].mean()
        new_row["rolling_mean_30"] = history[target_col].iloc[-30:].mean()

        history = pd.concat([history, new_row])

    # -----------------------------
    # 4. Create forecast series
    # -----------------------------
    future_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    forecast_series = pd.Series(predictions, index=future_index)

    # -----------------------------
    # 5. Plot forecast
    # -----------------------------
    plt.figure(figsize=(12, 5))

    df[target_col].plot(label="Historical Data")
    forecast_series.plot(label="Random Forest Recursive Forecast")

    plt.legend()
    plt.title(f"Random Forest Recursive Multi-Step Forecast ({steps} Days)")
    plt.ylabel("Temperature (°C)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"recursive_forecast_{steps}_days.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nRandom Forest Recursive Forecast:")
    print(forecast_series.head(10))

    return forecast_series


if __name__ == "__main__":
    recursive_forecast(steps=30)

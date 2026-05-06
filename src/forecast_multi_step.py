import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from statsmodels.tsa.deterministic import DeterministicProcess

from query_data import fetch_all_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


def multi_step_forecast():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all_data()

    y = df["temperature_2m_mean"]

    # ------------------------
    # Build deterministic features
    # ------------------------
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        drop=True
    )

    X = dp.in_sample()

    # -----------------------
    # Train model on ALL data
    # -----------------------
    model = LinearRegression()
    model.fit(X,y)

    # ------------------------
    # Forecast Horizon (30 days)
    # ------------------------
    steps=30

    X_future = dp.out_of_sample(steps=steps)

    y_forecast = model.predict(X_future)

    # -------------------------
    # Create future index
    # -------------------------
    future_index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    y_forecast = pd.Series(y_forecast, index=future_index)

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(12,5))

    y.plot(label="Historical Data")
    y_forecast.plot(label="30-day Forecast")

    plt.legend()
    plt.title("Multi-step Forecast (30 days)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_step_forecast_30_days.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    multi_step_forecast()

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.deterministic import DeterministicProcess

from query_data import fetch_all_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


def train_deterministic_model():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all_data()

    # Target
    y = df["temperature_2m_mean"]

    # ----------------------------
    # Build Deterministic Process
    # ----------------------------
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,        # linear Trend
        seasonal=True,  # monthly seasonality
        drop=True
    )

    X = dp.in_sample()

    # -----------------------
    # Train / Test Split (IMPORTANT)
    # -----------------------
    split_date = "2024-01-01"

    X_train = X[:split_date]
    X_test = X[split_date:]

    y_train = y[:split_date]
    y_test = y[split_date:]

    # -------------------------
    # Train model
    # -------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --------------------------
    # Predictions
    # --------------------------
    y_pred = model.predict(X_test)

    # --------------------------
    # Evaluation
    # --------------------------
    mae = mean_absolute_error(y_test, y_pred)

    print("\nDeterministic model MAE:", round(mae, 3))

    # -------------------------
    # Plot results
    # -------------------------
    plt.figure(figsize=(12,5))

    y_train.plot(label="Train")
    y_test.plot(label="Test")
    pd.Series(y_pred, index=y_test.index).plot(label="Prediction")

    plt.legend()
    plt.title("Deterministic Model Forecast")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deterministic_model_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_deterministic_model()

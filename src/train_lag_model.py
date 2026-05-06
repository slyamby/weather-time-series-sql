import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from query_data import fetch_all_data
from features2 import create_time_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


def train_lag_model():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_all_data()

    df = create_time_features(df)

    # Target
    y = df["temperature_2m_mean"]

    # Features (lags + cyclic + rolling)
    X = df.drop(columns=["temperature_2m_mean", "city"])

    # ------------------------
    # Train/Test Split
    # ------------------------
    split_date = "2024-01-01"

    X_train = X[:split_date]
    X_test = X[split_date:]

    y_train = y[:split_date]
    y_test = y[split_date:]

    # --------------------------
    # Train model
    # -------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -------------------------
    # Predict
    # -------------------------
    y_pred = model.predict(X_test)

    # ------------------------
    # Evaluate
    # ------------------------
    mae = mean_absolute_error(y_test, y_pred)

    print("\nLag Model MAE: ", round(mae, 3))

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(12,5))

    y_train.plot(label="Train")
    y_test.plot(label="Test")
    pd.Series(y_pred, index=y_test.index).plot(label="Prediction")

    plt.legend()
    plt.title("Lag Model Forecast")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_model_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_lag_model()

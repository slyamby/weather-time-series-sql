from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from query_data import fetch_all_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "charts"


def save_current_figure(filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(df):
    """
    Plot raw time series
    """
    plt.figure(figsize=(12,5))
    df["temperature_2m_mean"].plot()
    plt.title("Daily Temperature (Accra)")
    plt.ylabel("Temperature (°C)")
    save_current_figure("daily_temperature_accra.png")



def plot_rolling_average(df):
    """
    Plot rolling averages
    """
    plt.figure(figsize=(12,5))

    df["temperature_2m_mean"].plot(label="Original", alpha=0.5)
    df["temperature_2m_mean"].rolling(30).mean().plot(label="30-day avg")

    plt.title("Rolling Average Temperature")
    plt.legend()
    save_current_figure("rolling_average_temperature.png")


if __name__ == "__main__":
    df = fetch_all_data()

    plot_time_series(df)
    plot_rolling_average(df)

import sqlite3
from pathlib import Path

DB_PATH = Path("data/weather.db")


def get_connection():
    """
    Create and return a SQLite database connection 
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn
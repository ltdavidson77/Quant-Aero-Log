# ==========================
# api_clients/alpaca_feed.py
# ==========================
# Retrieves historical and live data from Alpaca API (stocks, ETFs).

import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# ----------------------------
# Alpaca Key Configuration
# ----------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")  # Insert your key here
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

# ----------------------------
# Fetch Historical Bars
# ----------------------------
def fetch_alpaca_data(symbol="AAPL", timeframe="1Min", limit=1000):
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "limit": limit,
        "adjustment": "raw"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Alpaca API Error: {response.text}")
    data = response.json()["bars"]

    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    df.rename(columns={"c": "price"}, inplace=True)
    return df[["price"]]

# ----------------------------
# Example Manual Test
# ----------------------------
if __name__ == "__main__":
    print("[ALPACA] Fetching test data for AAPL...")
    df = fetch_alpaca_data("AAPL", limit=50)
    print(df.head())

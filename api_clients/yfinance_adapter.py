# ==========================
# api_clients/yfinance_adapter.py
# ==========================
# Fallback data source using Yahoo Finance.

import yfinance as yf
import pandas as pd
from datetime import datetime

# ----------------------------
# Fetch Data from Yahoo Finance
# ----------------------------
def fetch_yfinance_data(symbol="AAPL", interval="1m", period="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, period=period)
    if df.empty:
        raise Exception(f"Yahoo Finance returned no data for {symbol}")

    df.index = pd.to_datetime(df.index)
    df.rename(columns={"Close": "price"}, inplace=True)
    return df[["price"]]

# ----------------------------
# Example Manual Test
# ----------------------------
if __name__ == "__main__":
    print("[YFINANCE] Fetching test data for AAPL...")
    df = fetch_yfinance_data("AAPL")
    print(df.head())

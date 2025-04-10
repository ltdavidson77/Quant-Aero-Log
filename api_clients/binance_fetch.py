# ==========================
# api_clients/binance_fetch.py
# ==========================
# Fetches cryptocurrency price data from Binance API.

import requests
import pandas as pd
from datetime import datetime

# ----------------------------
# Binance Kline API Endpoint
# ----------------------------
BASE_URL = "https://api.binance.com/api/v3/klines"

# ----------------------------
# Fetch Historical Crypto Bars
# ----------------------------
def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1m", limit=1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Binance API error: {response.text}")

    raw = response.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df["price"] = df["close"].astype(float)
    return df[["price"]]

# ----------------------------
# Example Manual Test
# ----------------------------
if __name__ == "__main__":
    print("[BINANCE] Fetching test data for BTCUSDT...")
    df = fetch_binance_ohlcv("BTCUSDT", limit=10)
    print(df.head())

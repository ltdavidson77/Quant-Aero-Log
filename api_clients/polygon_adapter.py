# File: api_clients/polygon_adapter.py
# ------------------------------------
# Polygon.io adapter to fetch OHLCV price data

import requests
import pandas as pd
import os

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "your_polygon_api_key")
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

def fetch_polygon_data(ticker: str, timespan: str = "minute", multiplier: int = 1, limit: int = 100):
    url = f"{BASE_URL}/{ticker}/range/{multiplier}/{timespan}/2023-01-01/2023-01-02"
    params = {"apiKey": POLYGON_API_KEY, "limit": limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Polygon API error: {response.text}")

    data = response.json()
    if "results" not in data:
        raise ValueError("No results in Polygon API response")

    df = pd.DataFrame(data["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

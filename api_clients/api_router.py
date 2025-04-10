# ==========================
# api_clients/api_router.py
# ==========================
# Central router to switch between data sources.

from api_clients.alpaca_feed import fetch_alpaca_data
from api_clients.binance_fetch import fetch_binance_ohlcv
from api_clients.yfinance_adapter import fetch_yfinance_data

# ----------------------------
# Unified Data Fetch Interface
# ----------------------------
def fetch_price_data(source="yfinance", symbol="AAPL", **kwargs):
    source = source.lower()

    if source == "alpaca":
        return fetch_alpaca_data(symbol=symbol, **kwargs)
    elif source == "binance":
        return fetch_binance_ohlcv(symbol=symbol, **kwargs)
    elif source == "yfinance":
        return fetch_yfinance_data(symbol=symbol, **kwargs)
    else:
        raise ValueError(f"Unsupported data source: {source}")

# ----------------------------
# Example Manual Test
# ----------------------------
if __name__ == "__main__":
    print("[API ROUTER] Fetching sample data from Yahoo Finance...")
    df = fetch_price_data("yfinance", symbol="AAPL")
    print(df.head())

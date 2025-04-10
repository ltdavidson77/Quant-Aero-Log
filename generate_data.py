# ==========================
# generate_data.py
# ==========================
# Handles price series generation for simulation or live feed.
# Optionally integrates real-time APIs (e.g. Alpaca, Binance).

import os
import numpy as np
import pandas as pd
from datetime import datetime
from config_env import USE_LIVE_FEED, DEBUG_MODE

# Optional: Placeholders for live feed integrations
# from api_clients.alpaca_feed import fetch_alpaca_data
# from api_clients.binance_fetch import fetch_binance_data

# ------------------------------------
# Simulated Price Series Generator
# ------------------------------------
def generate_simulated_series(length=43200, seed=42, base_price=100.0):
    np.random.seed(seed)
    price = np.cumsum(np.random.normal(loc=0.001, scale=0.5, size=length)) + base_price
    rng = pd.date_range(start=datetime.utcnow(), periods=length, freq='min')
    return pd.DataFrame({'price': price}, index=rng)

# ------------------------------------
# Real-Time Fetch Stub (optional)
# ------------------------------------
def generate_live_series(symbol='AAPL', source='alpaca'):
    if source == 'alpaca':
        # return fetch_alpaca_data(symbol)
        raise NotImplementedError("Alpaca data fetch not implemented.")
    elif source == 'binance':
        # return fetch_binance_data(symbol)
        raise NotImplementedError("Binance data fetch not implemented.")
    else:
        raise ValueError("Unknown data source: {}".format(source))

# ------------------------------------
# Primary Interface Function
# ------------------------------------
def get_price_series(symbol='SIM', length=43200):
    if USE_LIVE_FEED:
        if DEBUG_MODE:
            print("[DATA] Using live feed for symbol:", symbol)
        return generate_live_series(symbol)
    else:
        if DEBUG_MODE:
            print("[DATA] Using simulated feed with length:", length)
        return generate_simulated_series(length=length)

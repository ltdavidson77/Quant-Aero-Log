# ==========================
# angles_metrics.py
# ==========================
# Computes angular-based trigonometric metrics used in volatility and signal analysis.

import numpy as np
import pandas as pd

# -------------------------------
# Angular Metric Computation
# -------------------------------
def compute_angles(df, column='price', window=10):
    """
    Computes key angular features:
    - theta: raw angle of price movement
    - phi: angle of smoothed price (moving average)
    - acc: angular acceleration (second derivative)
    - vol: rolling volatility (standard deviation)
    """
    ma = df[column].rolling(window).mean().fillna(method='bfill')
    theta = np.arctan(np.gradient(df[column]))
    phi = np.arctan(np.gradient(ma))
    acc = np.gradient(np.gradient(df[column]))
    vol = df[column].rolling(window).std().fillna(method='bfill')
    return theta, phi, acc, vol

# -------------------------------
# Combined Angular Metrics as DataFrame
# -------------------------------
def compute_angle_dataframe(df, column='price', window=10):
    theta, phi, acc, vol = compute_angles(df, column, window)
    return pd.DataFrame({
        'theta': theta,
        'phi': phi,
        'acc': acc,
        'vol': vol
    }, index=df.index)

# -------------------------------
# Debugging Utility
# -------------------------------
def print_angular_summary(df):
    print("[METRICS] Angular Metric Summary:")
    print(df.describe())

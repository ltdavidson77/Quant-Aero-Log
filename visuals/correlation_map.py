# ==========================
# visuals/correlation_map.py
# ==========================
# Generate correlation matrix heatmap for signal features.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Plot Correlation Matrix
# ----------------------------------------
def plot_correlation_matrix(df, title="Signal Correlation Map", figsize=(12, 8)):
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# Example Manual Test
# ----------------------------------------
if __name__ == "__main__":
    from multi_timeframe import compute_multi_timeframe_signals
    from generate_data import get_price_series

    print("[VISUAL] Generating signal correlation matrix...")
    df = get_price_series()
    signal_df = compute_multi_timeframe_signals(df)
    plot_correlation_matrix(signal_df)

# ==========================
# plot_signals.py
# ==========================
# Visualization of price and multi-timeframe signal overlays.

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------
# Signal Plotting with Timeframe Overlays
# -----------------------------------------------------
def plot_timeframe_signals(df, signal_df, column='price', timeframes=['log_10m', 'log_60m', 'log_1440m']):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df[column], label='Price', color='black', alpha=0.3)
    for tf in timeframes:
        if tf in signal_df.columns:
            aligned = signal_df[tf].reindex(df.index, method='bfill')
            plt.plot(df.index, aligned, label=f'Signal: {tf}')
    plt.title('Angular Logarithmic Signals Across Timeframes')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------
# Optional Zoomed-In Window
# -----------------------------------------------------
def plot_zoomed_window(df, signal_df, start_time, end_time, timeframes=['log_10m', 'log_60m']):
    zoom_df = df[(df.index >= start_time) & (df.index <= end_time)]
    zoom_sig = signal_df[(signal_df.index >= start_time) & (signal_df.index <= end_time)]
    plot_timeframe_signals(zoom_df, zoom_sig, timeframes=timeframes)

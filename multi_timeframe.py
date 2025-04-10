# ==========================
# multi_timeframe.py
# ==========================
# Constructs multi-resolution angular signals using custom resolutions.

import pandas as pd
from angles_metrics import compute_angles
from log_signal import compute_log_signal
from alg_signal import compute_alg_signal

# ----------------------------------------------
# Multi-Timeframe Signal Generator
# ----------------------------------------------
def compute_multi_timeframe_signals(df, column='price', resolutions=[1,5,10,60,180,300,600,1440,2880,10080,43200]):
    results = {}
    for r in resolutions:
        resampled = df[column].resample(f'{r}min').last().interpolate(method='linear')
        sub_df = pd.DataFrame({column: resampled})
        theta, phi, acc, vol = compute_angles(sub_df, column)
        log_sig = compute_log_signal(theta, phi, acc, vol)
        alg_sig = compute_alg_signal(theta, phi, acc, vol)
        results[f'log_{r}m'] = pd.Series(log_sig, index=sub_df.index)
        results[f'alg_{r}m'] = pd.Series(alg_sig, index=sub_df.index)
    return pd.DataFrame(results)

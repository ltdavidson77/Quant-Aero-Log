# ==========================
# label_generator.py
# ==========================
# Creates multi-horizon labels based on future price returns for classification tasks.

import numpy as np
import pandas as pd

# --------------------------------------------------
# Label Generator for Future Returns
# --------------------------------------------------
def generate_signal_labels(df, column='price', horizon=15, threshold=0.3):
    """
    Label values:
    1 => price increased above threshold
    0 => price changed within threshold band
    -1 => price dropped below threshold
    """
    future_return = df[column].shift(-horizon) - df[column]
    label = np.where(future_return > threshold, 1,
             np.where(future_return < -threshold, -1, 0))
    return pd.Series(label, index=df.index, name=f'label_{horizon}')

# --------------------------------------------------
# Multi-Horizon Label Set Builder
# --------------------------------------------------
def generate_multi_horizon_labels(df, column='price', horizons=[5, 15, 60, 1440]):
    label_df = pd.DataFrame(index=df.index)
    for h in horizons:
        label_df[f'label_{h}'] = generate_signal_labels(df, column=column, horizon=h)
    return label_df

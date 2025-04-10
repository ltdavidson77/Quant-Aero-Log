# ==========================
#   log_signal.py 
# ==========================
# Constructs multi-layered logarithmic signals using angular input metrics.

import numpy as np

# ----------------------------------------
# Logarithmic Signal Construction Function
# ----------------------------------------
def compute_log_signal(theta, phi, acc, vol, alpha=1.0, beta=1.0, gamma=0.5, delta=0.5):
    """
    Constructs log-based signal using:
    - theta: angular momentum
    - phi: smoothed angular direction
    - acc: angular acceleration
    - vol: volatility index
    Each component is weighted by hyperparameters.
    """
    L1 = np.log(1 + alpha * np.sin(theta)**2 - beta * np.cos(phi)**2)
    L2 = np.log(1 + gamma * vol**2 * np.log(1 + np.abs(np.sin(theta - phi))))
    L3 = np.log(1 + delta * np.abs(acc) * np.sin(theta)**2)
    return np.log(1 + L1 + L2 + L3)

# ----------------------------------------
# Batch Construction from DataFrame Input
# ----------------------------------------
def build_log_signal_from_df(df):
    return compute_log_signal(
        df['theta'].values,
        df['phi'].values,
        df['acc'].values,
        df['vol'].values
    )

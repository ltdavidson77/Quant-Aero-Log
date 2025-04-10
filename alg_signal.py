# ==========================
# alg_signal.py
# ==========================
# Constructs algorithmic signal layers using angular and volatility-based metrics.

import numpy as np

# ------------------------------------------
# Algorithmic Signal Construction Function
# ------------------------------------------
def compute_alg_signal(theta, phi, acc, vol, a1=0.8, b1=0.6, g1=0.4):
    """
    Computes non-logarithmic signal using:
    - a1: weight on angular interference
    - b1: weight on acceleration*direction
    - g1: weight on volatility-amplitude cross product
    """
    A1 = a1 * np.sin(theta + 0.5 * phi)**2
    A2 = b1 * (acc**2) * np.cos(phi)**2
    A3 = g1 * vol * np.abs(np.sin(phi) * np.cos(theta))
    return A1 + A2 + A3

# ------------------------------------------
# Batch Construction from DataFrame Input
# ------------------------------------------
def build_alg_signal_from_df(df):
    return compute_alg_signal(
        df['theta'].values,
        df['phi'].values,
        df['acc'].values,
        df['vol'].values
    )

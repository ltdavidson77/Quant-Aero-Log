# ==========================
# compute_analytics.py
# ==========================
# Analytics computation functions for signal processing.

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def compute_angles(
    data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Compute angle-based metrics from price data.
    
    Args:
        data: DataFrame with price data
        window: Rolling window size
        threshold: Angle threshold for signal generation
        
    Returns:
        DataFrame with angle metrics
    """
    try:
        # Calculate price changes
        price_changes = data['Close'].diff()
        
        # Calculate angles using arctan
        angles = np.arctan(price_changes / data['Close'].shift(1))
        
        # Create rolling angle metrics
        rolling_angles = angles.rolling(window=window).mean()
        angle_std = angles.rolling(window=window).std()
        
        # Generate signals based on angle thresholds
        signals = pd.Series(0, index=data.index)
        signals[rolling_angles > threshold] = 1
        signals[rolling_angles < -threshold] = -1
        
        return pd.DataFrame({
            'angles': angles,
            'rolling_angles': rolling_angles,
            'angle_std': angle_std,
            'angle_signals': signals
        })
    except Exception as e:
        logger.error(f"Error computing angles: {str(e)}")
        return pd.DataFrame()

def compute_log_signal(
    data: pd.DataFrame,
    window: int = 20,
    smoothing: float = 0.1
) -> pd.DataFrame:
    """
    Compute log-based signals from price data.
    
    Args:
        data: DataFrame with price data
        window: Rolling window size
        smoothing: Smoothing factor for signal
        
    Returns:
        DataFrame with log signals
    """
    try:
        # Calculate log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate rolling statistics
        rolling_mean = log_returns.rolling(window=window).mean()
        rolling_std = log_returns.rolling(window=window).std()
        
        # Generate normalized signals
        signals = (log_returns - rolling_mean) / rolling_std
        signals = signals.ewm(alpha=smoothing).mean()
        
        return pd.DataFrame({
            'log_returns': log_returns,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'log_signals': signals
        })
    except Exception as e:
        logger.error(f"Error computing log signals: {str(e)}")
        return pd.DataFrame()

def compute_alg_signal(
    data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute algorithmic trading signals from price data.
    
    Args:
        data: DataFrame with price data
        window: Rolling window size
        threshold: Signal threshold
        
    Returns:
        DataFrame with algorithmic signals
    """
    try:
        # Calculate price momentum
        momentum = data['Close'].pct_change(periods=window)
        
        # Calculate volatility
        volatility = data['Close'].rolling(window=window).std() / data['Close']
        
        # Calculate trend strength
        trend = data['Close'].rolling(window=window).mean()
        trend_strength = (data['Close'] - trend) / trend
        
        # Generate composite signals
        signals = pd.Series(0, index=data.index)
        signals[(momentum > threshold) & (trend_strength > 0)] = 1
        signals[(momentum < -threshold) & (trend_strength < 0)] = -1
        
        return pd.DataFrame({
            'momentum': momentum,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'alg_signals': signals
        })
    except Exception as e:
        logger.error(f"Error computing algorithmic signals: {str(e)}")
        return pd.DataFrame() 
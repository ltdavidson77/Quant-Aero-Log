# ==========================
# signal_labeler.py
# ==========================
# Handles generation of multi-horizon labels for time series data.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from config_env import LABEL_HORIZONS

logger = logging.getLogger(__name__)

def generate_multi_horizon_labels(
    df: pd.DataFrame,
    price_col: str = 'Close',
    horizons: Optional[List[int]] = None,
    threshold: float = 0.02
) -> pd.DataFrame:
    """
    Generate multi-horizon labels for time series data.
    
    Args:
        df: DataFrame containing price data
        price_col: Column name for price data
        horizons: List of horizon periods in minutes
        threshold: Minimum price change threshold for labeling
        
    Returns:
        DataFrame with generated labels
    """
    try:
        # Use default horizons if none provided
        horizons = horizons or LABEL_HORIZONS
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Generate labels for each horizon
        for horizon in horizons:
            # Calculate future returns
            future_prices = df[price_col].shift(-horizon)
            returns = (future_prices - df[price_col]) / df[price_col]
            
            # Generate binary labels
            labels = np.zeros(len(df))
            labels[returns > threshold] = 1  # Up movement
            labels[returns < -threshold] = -1  # Down movement
            
            # Store labels
            result[f'label_{horizon}m'] = labels
            
        # Add metadata
        result['label_generation_time'] = datetime.utcnow()
        result['label_threshold'] = threshold
        
        logger.info(f"Generated labels for horizons: {horizons}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating labels: {str(e)}")
        raise

def calculate_label_quality(
    labels: pd.DataFrame,
    actual_returns: pd.Series,
    horizon: int
) -> Dict[str, float]:
    """
    Calculate quality metrics for generated labels.
    
    Args:
        labels: DataFrame containing generated labels
        actual_returns: Series of actual returns
        horizon: Label horizon in minutes
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        # Get labels for specified horizon
        horizon_labels = labels[f'label_{horizon}m']
        
        # Calculate metrics
        accuracy = np.mean(
            (horizon_labels > 0) == (actual_returns > 0)
        )
        
        precision = np.mean(
            actual_returns[horizon_labels > 0] > 0
        ) if np.any(horizon_labels > 0) else 0
        
        recall = np.mean(
            horizon_labels[actual_returns > 0] > 0
        ) if np.any(actual_returns > 0) else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'horizon': horizon
        }
        
    except Exception as e:
        logger.error(f"Error calculating label quality: {str(e)}")
        raise

def validate_labels(
    labels: pd.DataFrame,
    min_samples: int = 100,
    min_class_balance: float = 0.1
) -> Tuple[bool, str]:
    """
    Validate generated labels for quality and balance.
    
    Args:
        labels: DataFrame containing generated labels
        min_samples: Minimum number of samples required
        min_class_balance: Minimum ratio of minority class
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check sample size
        if len(labels) < min_samples:
            return False, f"Insufficient samples: {len(labels)} < {min_samples}"
            
        # Check class balance for each horizon
        for col in labels.columns:
            if col.startswith('label_'):
                class_counts = labels[col].value_counts()
                minority_ratio = min(class_counts) / max(class_counts)
                
                if minority_ratio < min_class_balance:
                    return False, f"Poor class balance in {col}: {minority_ratio:.2f} < {min_class_balance}"
                    
        return True, "Labels validated successfully"
        
    except Exception as e:
        logger.error(f"Error validating labels: {str(e)}")
        raise 
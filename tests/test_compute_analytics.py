# ==========================
# tests/test_compute_analytics.py
# ==========================
# Test cases for analytics computation functions.

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from compute_analytics import (
    compute_angles,
    compute_log_signal,
    compute_alg_signal
)

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5T')
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Open': prices * 0.999,
        'High': prices * 1.001,
        'Low': prices * 0.998,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

def test_compute_angles(sample_price_data):
    """Test angle computation functionality."""
    # Test with default parameters
    result = compute_angles(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    assert 'angles' in result.columns
    assert 'rolling_angles' in result.columns
    assert 'angle_std' in result.columns
    assert 'angle_signals' in result.columns
    
    # Test signal generation
    assert result['angle_signals'].isin([-1, 0, 1]).all()
    
    # Test with custom parameters
    custom_result = compute_angles(
        sample_price_data,
        window=30,
        threshold=0.15
    )
    assert len(custom_result) == len(sample_price_data)

def test_compute_log_signal(sample_price_data):
    """Test log signal computation functionality."""
    # Test with default parameters
    result = compute_log_signal(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    assert 'log_returns' in result.columns
    assert 'rolling_mean' in result.columns
    assert 'rolling_std' in result.columns
    assert 'log_signals' in result.columns
    
    # Test signal normalization
    assert abs(result['log_signals'].mean()) < 0.1  # Should be roughly centered
    
    # Test with custom parameters
    custom_result = compute_log_signal(
        sample_price_data,
        window=30,
        smoothing=0.2
    )
    assert len(custom_result) == len(sample_price_data)

def test_compute_alg_signal(sample_price_data):
    """Test algorithmic signal computation functionality."""
    # Test with default parameters
    result = compute_alg_signal(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    assert 'momentum' in result.columns
    assert 'volatility' in result.columns
    assert 'trend_strength' in result.columns
    assert 'alg_signals' in result.columns
    
    # Test signal generation
    assert result['alg_signals'].isin([-1, 0, 1]).all()
    
    # Test with custom parameters
    custom_result = compute_alg_signal(
        sample_price_data,
        window=30,
        threshold=0.7
    )
    assert len(custom_result) == len(sample_price_data)

def test_error_handling():
    """Test error handling in analytics computation."""
    # Test with invalid input
    with pytest.raises(Exception):
        compute_angles(pd.DataFrame())
    
    # Test with missing required columns
    invalid_df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(Exception):
        compute_log_signal(invalid_df)
    
    # Test with non-numeric data
    invalid_df = pd.DataFrame({'Close': ['a', 'b', 'c']})
    with pytest.raises(Exception):
        compute_alg_signal(invalid_df)

def test_edge_cases(sample_price_data):
    """Test edge cases in analytics computation."""
    # Test with single row
    single_row = sample_price_data.iloc[:1]
    result = compute_angles(single_row)
    assert len(result) == 1
    
    # Test with constant prices
    constant_prices = sample_price_data.copy()
    constant_prices['Close'] = 100.0
    result = compute_log_signal(constant_prices)
    assert result['log_returns'].iloc[1:].sum() == 0
    
    # Test with zero prices
    zero_prices = sample_price_data.copy()
    zero_prices['Close'] = 0.0
    with pytest.raises(Exception):
        compute_alg_signal(zero_prices) 
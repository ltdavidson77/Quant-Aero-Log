# ==========================
# tests/test_signal_labeler.py
# ==========================
# Test cases for signal labeling functions.

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from signal_labeler import (
    generate_multi_horizon_labels,
    calculate_label_quality,
    validate_labels
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

def test_generate_multi_horizon_labels(sample_price_data):
    """Test multi-horizon label generation."""
    # Test with default parameters
    result = generate_multi_horizon_labels(sample_price_data)
    assert isinstance(result, pd.DataFrame)
    assert 'label_generation_time' in result.columns
    assert 'label_threshold' in result.columns
    
    # Test with custom horizons
    custom_horizons = [5, 15, 30]
    custom_result = generate_multi_horizon_labels(
        sample_price_data,
        horizons=custom_horizons
    )
    for horizon in custom_horizons:
        assert f'label_{horizon}m' in custom_result.columns
    
    # Test label values
    for col in custom_result.columns:
        if col.startswith('label_'):
            assert custom_result[col].isin([-1, 0, 1]).all()

def test_calculate_label_quality(sample_price_data):
    """Test label quality calculation."""
    # Generate labels
    labels = generate_multi_horizon_labels(sample_price_data)
    
    # Create sample actual returns
    actual_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(sample_price_data)),
        index=sample_price_data.index
    )
    
    # Test quality calculation
    quality = calculate_label_quality(labels, actual_returns, 5)
    assert isinstance(quality, dict)
    assert 'accuracy' in quality
    assert 'precision' in quality
    assert 'recall' in quality
    assert 'f1_score' in quality
    assert 'horizon' in quality
    
    # Test metric ranges
    assert 0 <= quality['accuracy'] <= 1
    assert 0 <= quality['precision'] <= 1
    assert 0 <= quality['recall'] <= 1
    assert 0 <= quality['f1_score'] <= 1

def test_validate_labels(sample_price_data):
    """Test label validation."""
    # Generate valid labels
    labels = generate_multi_horizon_labels(sample_price_data)
    
    # Test validation with good data
    is_valid, message = validate_labels(labels)
    assert is_valid
    assert "successfully" in message
    
    # Test with insufficient samples
    small_labels = labels.iloc[:50]
    is_valid, message = validate_labels(small_labels, min_samples=100)
    assert not is_valid
    assert "Insufficient samples" in message
    
    # Test with poor class balance
    imbalanced_labels = labels.copy()
    imbalanced_labels['label_5m'] = 1  # All positive labels
    is_valid, message = validate_labels(imbalanced_labels)
    assert not is_valid
    assert "Poor class balance" in message

def test_error_handling():
    """Test error handling in label generation."""
    # Test with empty DataFrame
    with pytest.raises(Exception):
        generate_multi_horizon_labels(pd.DataFrame())
    
    # Test with invalid price column
    invalid_df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(Exception):
        generate_multi_horizon_labels(invalid_df)
    
    # Test with non-numeric data
    invalid_df = pd.DataFrame({'Close': ['a', 'b', 'c']})
    with pytest.raises(Exception):
        generate_multi_horizon_labels(invalid_df)

def test_edge_cases(sample_price_data):
    """Test edge cases in label generation."""
    # Test with single row
    single_row = sample_price_data.iloc[:1]
    result = generate_multi_horizon_labels(single_row)
    assert len(result) == 1
    
    # Test with constant prices
    constant_prices = sample_price_data.copy()
    constant_prices['Close'] = 100.0
    result = generate_multi_horizon_labels(constant_prices)
    assert result['label_5m'].sum() == 0  # No price movement
    
    # Test with zero prices
    zero_prices = sample_price_data.copy()
    zero_prices['Close'] = 0.0
    with pytest.raises(Exception):
        generate_multi_horizon_labels(zero_prices) 
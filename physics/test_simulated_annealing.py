import pytest
import numpy as np
import cupy as cp
from physics.simulated_annealing import Algorithm
import logging

class MockAnalysisAPI:
    def __init__(self):
        self.cache = {}
        
    def get(self, key, default=None):
        return self.cache.get(key, default)
        
    def set(self, key, value):
        self.cache[key] = value

@pytest.fixture
def api():
    return MockAnalysisAPI()

@pytest.fixture
def algorithm():
    return Algorithm()

@pytest.fixture
def sample_data():
    return {
        "AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000},
        "MSFT": {"Adj Close": 250.0, "Vol_1d": 800000, "Volume": 4000000},
        "GOOGL": {"Adj Close": 2000.0, "Vol_1d": 500000, "Volume": 2000000},
        "AMZN": {"Adj Close": 3000.0, "Vol_1d": 600000, "Volume": 3000000},
        "META": {"Adj Close": 180.0, "Vol_1d": 700000, "Volume": 3500000}
    }

@pytest.fixture
def sample_mesh_results():
    return {
        "centrality": {
            "AAPL": 0.8,
            "MSFT": 0.7,
            "GOOGL": 0.6,
            "AMZN": 0.5,
            "META": 0.4
        }
    }

def test_algorithm_initialization(algorithm):
    """Test algorithm initialization and configuration."""
    config = algorithm.get_default_config()
    assert isinstance(config, dict)
    assert "max_iter" in config
    assert "temp" in config
    assert "cooling_rate" in config
    assert "metric" in config
    
    metrics = algorithm.get_supported_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert "Adj Close" in metrics
    assert "Vol_1d" in metrics
    assert "Volume" in metrics
    
    version = algorithm.get_version()
    assert isinstance(version, str)
    assert version == "v2.0"

def test_run_with_valid_data(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with valid data."""
    api.set("mesh_results.1d", sample_mesh_results)
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    assert isinstance(result, dict)
    assert "sa_portfolio" in result
    assert "sa_score" in result
    
    assert isinstance(result["sa_portfolio"], list)
    assert len(result["sa_portfolio"]) > 0
    assert all(stock in sample_data for stock in result["sa_portfolio"])
    
    assert isinstance(result["sa_score"], float)
    assert result["sa_score"] > 0

def test_run_with_insufficient_data(algorithm, api):
    """Test running the algorithm with insufficient data."""
    # Test with empty data
    result = algorithm.run(api, {}, "1d", algorithm.get_default_config())
    assert result == {"sa_portfolio": [], "sa_score": 0.0}
    
    # Test with single stock
    single_stock = {"AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000}}
    result = algorithm.run(api, single_stock, "1d", algorithm.get_default_config())
    assert result == {"sa_portfolio": [], "sa_score": 0.0}

def test_run_with_custom_config(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with custom configuration."""
    api.set("mesh_results.1d", sample_mesh_results)
    config = algorithm.get_default_config()
    config.update({
        "max_iter": 50,
        "temp": 500.0,
        "cooling_rate": 0.9,
        "metric": "Volume"
    })
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["sa_portfolio"]) > 0
    assert result["sa_score"] > 0

def test_result_caching(algorithm, api, sample_data, sample_mesh_results):
    """Test that results are properly cached."""
    api.set("mesh_results.1d", sample_mesh_results)
    algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    cached_result = api.get("algo_cache.simulated_annealing.1d")
    assert cached_result is not None
    assert "sa_portfolio" in cached_result
    assert "sa_score" in cached_result

def test_performance_large_dataset(algorithm, api):
    """Test performance with a large dataset."""
    large_data = {f"STOCK_{i}": {
        "Adj Close": 100.0,
        "Vol_1d": 1000000,
        "Volume": 5000000
    } for i in range(100)}
    large_mesh = {"centrality": {f"STOCK_{i}": 0.5 for i in range(100)}}
    api.set("mesh_results.1d", large_mesh)
    result = algorithm.run(api, large_data, "1d", algorithm.get_default_config())
    assert len(result["sa_portfolio"]) > 0
    assert result["sa_score"] > 0

def test_error_handling(algorithm, api, sample_data):
    """Test error handling with invalid data."""
    # Test with invalid mesh results
    api.set("mesh_results.1d", {"invalid": "data"})
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    assert result == {"sa_portfolio": [], "sa_score": 0.0}

def test_mesh_integration(algorithm, api, sample_data, sample_mesh_results):
    """Test integration with mesh topology."""
    api.set("mesh_results.1d", sample_mesh_results)
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    # Verify that high centrality stocks are more likely to be selected
    high_centrality_stocks = [stock for stock, centrality in sample_mesh_results["centrality"].items() 
                            if centrality > 0.6]
    selected_high_centrality = [stock for stock in result["sa_portfolio"] 
                              if stock in high_centrality_stocks]
    assert len(selected_high_centrality) > 0

def test_metric_selection(algorithm, api, sample_data, sample_mesh_results):
    """Test different metric selections."""
    api.set("mesh_results.1d", sample_mesh_results)
    for metric in algorithm.get_supported_metrics():
        config = algorithm.get_default_config()
        config["metric"] = metric
        result = algorithm.run(api, sample_data, "1d", config)
        assert len(result["sa_portfolio"]) > 0
        assert result["sa_score"] > 0

def test_annealing_parameters(algorithm, api, sample_data, sample_mesh_results):
    """Test different annealing parameters."""
    api.set("mesh_results.1d", sample_mesh_results)
    test_params = [
        {"max_iter": 50, "temp": 500.0, "cooling_rate": 0.9},
        {"max_iter": 200, "temp": 2000.0, "cooling_rate": 0.99},
        {"max_iter": 100, "temp": 1000.0, "cooling_rate": 0.95}
    ]
    
    for params in test_params:
        config = algorithm.get_default_config()
        config.update(params)
        result = algorithm.run(api, sample_data, "1d", config)
        assert len(result["sa_portfolio"]) > 0
        assert result["sa_score"] > 0 
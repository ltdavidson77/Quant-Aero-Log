import pytest
import numpy as np
import cupy as cp
from algorithms.dijkstra_correlation import Algorithm
import networkx as nx

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
        "AAPL": {"Adj Close": 150.0, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0},
        "MSFT": {"Adj Close": 250.0, "Volume": 4000000, "High": 255.0, "Low": 245.0, "Open": 248.0, "Close": 250.0},
        "GOOGL": {"Adj Close": 2000.0, "Volume": 2000000, "High": 2010.0, "Low": 1990.0, "Open": 2005.0, "Close": 2000.0},
        "AMZN": {"Adj Close": 3000.0, "Volume": 3000000, "High": 3010.0, "Low": 2990.0, "Open": 3005.0, "Close": 3000.0},
        "META": {"Adj Close": 180.0, "Volume": 3500000, "High": 185.0, "Low": 175.0, "Open": 178.0, "Close": 180.0}
    }

@pytest.fixture
def sample_mesh_results():
    return {
        "centrality": {
            "AAPL": 0.8,
            "MSFT": 0.6,
            "GOOGL": 0.4,
            "AMZN": 0.5,
            "META": 0.7
        }
    }

def test_algorithm_initialization(algorithm):
    """Test algorithm initialization and configuration."""
    config = algorithm.get_default_config()
    assert isinstance(config, dict)
    assert "base_metric" in config
    assert "correlation_threshold" in config
    assert "use_advanced_metrics" in config
    assert "pca_components" in config
    assert "correlation_method" in config
    assert "centrality_weight" in config
    assert "volatility_weight" in config
    assert "momentum_weight" in config
    assert "liquidity_weight" in config
    assert "max_workers" in config
    assert "use_gpu" in config
    assert "optimization_level" in config
    
    metrics = algorithm.get_supported_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert "Adj Close" in metrics
    assert "Volume" in metrics
    assert "High" in metrics
    assert "Low" in metrics
    assert "Open" in metrics
    assert "Close" in metrics
    
    version = algorithm.get_version()
    assert isinstance(version, str)
    assert version == "v3.0"

def test_advanced_metrics_computation(algorithm, sample_data):
    """Test computation of advanced financial metrics."""
    metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    assert isinstance(metrics, dict)
    for stock in sample_data:
        assert stock in metrics
        assert "volatility" in metrics[stock]
        assert "momentum" in metrics[stock]
        assert "liquidity" in metrics[stock]
        assert "returns" in metrics[stock]

def test_correlation_matrix_computation(algorithm, sample_data):
    """Test correlation matrix computation with different methods."""
    for method in ["hybrid", "pearson", "spearman", "kendall"]:
        corr_matrix = algorithm._compute_correlation_matrix(sample_data, method)
        assert isinstance(corr_matrix, cp.ndarray)
        assert corr_matrix.shape == (len(sample_data), len(sample_data))
        assert not cp.isnan(corr_matrix).any()

def test_pca_application(algorithm, sample_data):
    """Test PCA application for dimensionality reduction."""
    n_components = 3
    pca_features = algorithm._apply_pca(sample_data, n_components)
    assert isinstance(pca_features, dict)
    for stock in sample_data:
        assert stock in pca_features
        assert len(pca_features[stock]) == n_components

def test_network_metrics_computation(algorithm, sample_data):
    """Test computation of network-based metrics."""
    corr_matrix = algorithm._compute_correlation_matrix(sample_data)
    network_metrics = algorithm._compute_network_metrics(corr_matrix, list(sample_data.keys()))
    assert isinstance(network_metrics, dict)
    assert "centrality" in network_metrics
    assert "clustering" in network_metrics
    for stock in sample_data:
        assert stock in network_metrics["centrality"]
        assert stock in network_metrics["clustering"]

def test_path_optimization(algorithm, sample_data):
    """Test path optimization with advanced techniques."""
    corr_matrix = algorithm._compute_correlation_matrix(sample_data)
    network_metrics = algorithm._compute_network_metrics(corr_matrix, list(sample_data.keys()))
    distances = 1 - cp.abs(cp.sin(corr_matrix))
    paths = algorithm._optimize_paths(distances, list(sample_data.keys()), network_metrics)
    assert isinstance(paths, dict)
    for stock, path in paths.items():
        assert stock in sample_data
        assert isinstance(path, list)
        assert len(path) > 0
        assert all(s in sample_data for s in path)

def test_run_with_valid_data(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with valid data."""
    api.set("mesh_results.1d", sample_mesh_results)
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    assert isinstance(result, dict)
    assert "correlation_paths" in result
    assert "execution_time" in result
    assert "network_metrics" in result
    assert "advanced_metrics" in result
    
    assert isinstance(result["correlation_paths"], dict)
    assert len(result["correlation_paths"]) > 0
    for path in result["correlation_paths"].values():
        assert isinstance(path, list)
        assert all(isinstance(stock, str) for stock in path)

def test_run_with_insufficient_data(algorithm, api):
    """Test running the algorithm with insufficient data."""
    # Test with empty data
    result = algorithm.run(api, {}, "1d", algorithm.get_default_config())
    assert result == {"correlation_paths": {}}
    
    # Test with single stock
    single_stock = {"AAPL": {"Adj Close": 150.0, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0}}
    result = algorithm.run(api, single_stock, "1d", algorithm.get_default_config())
    assert result == {"correlation_paths": {}}

def test_run_with_custom_config(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with custom configuration."""
    api.set("mesh_results.1d", sample_mesh_results)
    config = algorithm.get_default_config()
    config.update({
        "correlation_threshold": 0.9,
        "pca_components": 2,
        "correlation_method": "spearman",
        "centrality_weight": 0.4,
        "volatility_weight": 0.3,
        "momentum_weight": 0.2,
        "liquidity_weight": 0.1
    })
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["correlation_paths"]) > 0

def test_result_caching(algorithm, api, sample_data, sample_mesh_results):
    """Test that results are properly cached."""
    api.set("mesh_results.1d", sample_mesh_results)
    algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    cached_result = api.get("algo_cache.dijkstra_correlation.1d")
    assert cached_result is not None
    assert "correlation_paths" in cached_result
    assert "execution_time" in cached_result
    assert "network_metrics" in cached_result
    assert "advanced_metrics" in cached_result

def test_performance_large_dataset(algorithm, api):
    """Test performance with a large dataset."""
    large_data = {f"STOCK_{i}": {
        "Adj Close": 100.0, 
        "Volume": 5000000,
        "High": 105.0,
        "Low": 95.0,
        "Open": 98.0,
        "Close": 100.0
    } for i in range(100)}
    large_mesh = {"centrality": {f"STOCK_{i}": 0.5 for i in range(100)}}
    api.set("mesh_results.1d", large_mesh)
    result = algorithm.run(api, large_data, "1d", algorithm.get_default_config())
    assert len(result["correlation_paths"]) > 0
    assert result["execution_time"] < 10.0  # Should complete within 10 seconds

def test_error_handling(algorithm, api, sample_data):
    """Test error handling with invalid mesh results."""
    api.set("mesh_results.1d", {"invalid": "data"})
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    assert result == {"correlation_paths": {}}

def test_parallel_processing(algorithm, api, sample_data, sample_mesh_results):
    """Test parallel processing capabilities."""
    api.set("mesh_results.1d", sample_mesh_results)
    config = algorithm.get_default_config()
    config["max_workers"] = 2  # Reduce workers for testing
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["correlation_paths"]) > 0
    assert result["execution_time"] > 0 
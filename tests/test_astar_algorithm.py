import pytest
import numpy as np
import cupy as cp
from algorithms.astar_algorithm import Algorithm, OptimizationLevel
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
        "AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0},
        "MSFT": {"Adj Close": 250.0, "Vol_1d": 800000, "Volume": 4000000, "High": 255.0, "Low": 245.0, "Open": 248.0, "Close": 250.0},
        "GOOGL": {"Adj Close": 2000.0, "Vol_1d": 500000, "Volume": 2000000, "High": 2010.0, "Low": 1990.0, "Open": 2005.0, "Close": 2000.0},
        "AMZN": {"Adj Close": 3000.0, "Vol_1d": 600000, "Volume": 3000000, "High": 3010.0, "Low": 2990.0, "Open": 3005.0, "Close": 3000.0},
        "META": {"Adj Close": 180.0, "Vol_1d": 700000, "Volume": 3500000, "High": 185.0, "Low": 175.0, "Open": 178.0, "Close": 180.0}
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
    assert "goal_metric" in config
    assert "weight_factor" in config
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
    assert "heuristic_weight" in config
    assert "risk_adjustment" in config
    assert "market_regime" in config
    assert "time_horizon" in config
    assert "sector_weighting" in config
    assert "liquidity_threshold" in config
    assert "max_path_length" in config
    
    metrics = algorithm.get_supported_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert "Adj Close" in metrics
    assert "Vol_1d" in metrics
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
        assert "risk_adjusted_returns" in metrics[stock]
        assert "trend_strength" in metrics[stock]

def test_correlation_matrix_computation(algorithm, sample_data):
    """Test correlation matrix computation with different methods."""
    for method in ["hybrid", "pearson", "spearman", "kendall"]:
        corr_matrix = algorithm._compute_correlation_matrix(sample_data, method)
        assert isinstance(corr_matrix, cp.ndarray)
        assert corr_matrix.shape == (len(sample_data), len(sample_data))
        assert not cp.isnan(corr_matrix).any()

def test_heuristic_computation(algorithm, sample_data):
    """Test heuristic computation for A* algorithm."""
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    for stock in sample_data:
        heuristic = algorithm._compute_heuristic(stock, "Vol_1d", sample_data, advanced_metrics)
        assert isinstance(heuristic, float)
        assert 0 <= heuristic <= 1

def test_neighbor_selection(algorithm, sample_data):
    """Test neighbor selection in A* algorithm."""
    corr_matrix = algorithm._compute_correlation_matrix(sample_data)
    for stock in sample_data:
        neighbors = algorithm._get_neighbors(stock, sample_data, corr_matrix, list(sample_data.keys()))
        assert isinstance(neighbors, list)
        assert all(n in sample_data for n in neighbors)
        assert stock not in neighbors

def test_path_reconstruction(algorithm):
    """Test path reconstruction in A* algorithm."""
    from algorithms.astar_algorithm import Node
    node3 = Node("C", 3.0, 3.0)
    node2 = Node("B", 2.0, 2.0, node3)
    node1 = Node("A", 1.0, 1.0, node2)
    path = algorithm._reconstruct_path(node1)
    assert path == ["A", "B", "C"]

def test_a_star_search(algorithm, sample_data):
    """Test A* search implementation."""
    corr_matrix = algorithm._compute_correlation_matrix(sample_data)
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    for stock in sample_data:
        path = algorithm._a_star_search(stock, sample_data, corr_matrix, advanced_metrics)
        assert isinstance(path, list)
        assert len(path) > 0
        assert all(s in sample_data for s in path)
        assert path[0] == stock

def test_run_with_valid_data(algorithm, api, sample_data):
    """Test running the algorithm with valid data."""
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    assert isinstance(result, dict)
    assert "astar_priority" in result
    assert "astar_scores" in result
    assert "execution_time" in result
    assert "advanced_metrics" in result
    
    assert isinstance(result["astar_priority"], list)
    assert len(result["astar_priority"]) > 0
    assert all(isinstance(stock, str) for stock in result["astar_priority"])
    
    assert isinstance(result["astar_scores"], dict)
    assert len(result["astar_scores"]) > 0
    assert all(isinstance(score, float) for score in result["astar_scores"].values())

def test_run_with_insufficient_data(algorithm, api):
    """Test running the algorithm with insufficient data."""
    # Test with empty data
    result = algorithm.run(api, {}, "1d", algorithm.get_default_config())
    assert result == {"astar_priority": [], "astar_scores": []}
    
    # Test with single stock
    single_stock = {"AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0}}
    result = algorithm.run(api, single_stock, "1d", algorithm.get_default_config())
    assert result == {"astar_priority": [], "astar_scores": []}

def test_run_with_custom_config(algorithm, api, sample_data):
    """Test running the algorithm with custom configuration."""
    config = algorithm.get_default_config()
    config.update({
        "goal_metric": "Volume",
        "weight_factor": 0.7,
        "correlation_method": "spearman",
        "market_regime": "volatile",
        "risk_adjustment": True,
        "max_path_length": 5
    })
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["astar_priority"]) > 0

def test_result_caching(algorithm, api, sample_data):
    """Test that results are properly cached."""
    algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    cached_result = api.get("algo_cache.astar_prioritization.1d")
    assert cached_result is not None
    assert "astar_priority" in cached_result
    assert "astar_scores" in cached_result
    assert "execution_time" in cached_result
    assert "advanced_metrics" in cached_result

def test_performance_large_dataset(algorithm, api):
    """Test performance with a large dataset."""
    large_data = {f"STOCK_{i}": {
        "Adj Close": 100.0,
        "Vol_1d": 1000000,
        "Volume": 5000000,
        "High": 105.0,
        "Low": 95.0,
        "Open": 98.0,
        "Close": 100.0
    } for i in range(100)}
    result = algorithm.run(api, large_data, "1d", algorithm.get_default_config())
    assert len(result["astar_priority"]) > 0
    assert result["execution_time"] < 10.0  # Should complete within 10 seconds

def test_error_handling(algorithm, api, sample_data):
    """Test error handling with invalid data."""
    # Test with invalid metric
    config = algorithm.get_default_config()
    config["goal_metric"] = "InvalidMetric"
    result = algorithm.run(api, sample_data, "1d", config)
    assert result == {"astar_priority": [], "astar_scores": []}

def test_market_regime_adaptation(algorithm, api, sample_data):
    """Test adaptation to different market regimes."""
    for regime in ["normal", "volatile", "trending"]:
        config = algorithm.get_default_config()
        config["market_regime"] = regime
        result = algorithm.run(api, sample_data, "1d", config)
        assert len(result["astar_priority"]) > 0
        # Verify that the prioritization changes with market regime
        if regime == "volatile":
            assert any(metric["volatility"] > 0 for metric in result["advanced_metrics"].values())
        elif regime == "trending":
            assert any(metric["trend_strength"] > 0 for metric in result["advanced_metrics"].values())

def test_mesh_integration(algorithm, api, sample_data, sample_mesh_results):
    """Test integration with mesh topology."""
    api.set("mesh_results.1d", sample_mesh_results)
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    assert len(result["astar_priority"]) > 0
    # Verify that stocks with higher centrality tend to be prioritized
    high_centrality_stocks = [stock for stock, centrality in sample_mesh_results["centrality"].items() 
                            if centrality > 0.6]
    assert any(stock in result["astar_priority"][:3] for stock in high_centrality_stocks) 
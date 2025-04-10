import pytest
import numpy as np
import cupy as cp
from algorithms.kmeans_clustering import Algorithm, ClusteringMethod, DistanceMetric, ClusterStats
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
    assert "n_clusters" in config
    assert "metrics" in config
    assert "use_advanced_metrics" in config
    assert "pca_components" in config
    assert "clustering_method" in config
    assert "distance_metric" in config
    assert "centrality_weight" in config
    assert "volatility_weight" in config
    assert "momentum_weight" in config
    assert "liquidity_weight" in config
    assert "max_workers" in config
    assert "use_gpu" in config
    assert "normalization_method" in config
    assert "outlier_removal" in config
    assert "outlier_threshold" in config
    assert "min_cluster_size" in config
    assert "max_cluster_size" in config
    assert "silhouette_threshold" in config
    assert "market_regime" in config
    assert "time_horizon" in config
    assert "sector_weighting" in config
    assert "liquidity_threshold" in config
    
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

def test_feature_preparation(algorithm, sample_data, sample_mesh_results):
    """Test feature matrix preparation."""
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    X = algorithm._prepare_features(sample_data, advanced_metrics, sample_mesh_results["centrality"])
    
    assert isinstance(X, cp.ndarray)
    assert X.shape[0] == len(sample_data)
    assert not cp.isnan(X).any()

def test_outlier_removal(algorithm, sample_data):
    """Test outlier removal functionality."""
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    X = algorithm._prepare_features(sample_data, advanced_metrics, {})
    
    X_filtered, valid_indices = algorithm._remove_outliers(X, 3.0)
    assert isinstance(X_filtered, cp.ndarray)
    assert isinstance(valid_indices, list)
    assert len(valid_indices) <= len(sample_data)
    assert not cp.isnan(X_filtered).any()

def test_cluster_stats_computation(algorithm, sample_data, sample_mesh_results):
    """Test cluster statistics computation."""
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    cluster = list(sample_data.keys())[:3]  # Test with first 3 stocks
    
    stats = algorithm._compute_cluster_stats(cluster, sample_data, advanced_metrics, sample_mesh_results["centrality"])
    assert isinstance(stats, ClusterStats)
    assert stats.size == len(cluster)
    assert isinstance(stats.volatility, float)
    assert isinstance(stats.momentum, float)
    assert isinstance(stats.liquidity, float)
    assert isinstance(stats.centrality, float)
    assert isinstance(stats.silhouette, float)

def test_run_with_valid_data(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with valid data."""
    api.set("mesh_results.1d", sample_mesh_results)
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    assert isinstance(result, dict)
    assert "kmeans_clusters" in result
    assert "kmeans_centroids" in result
    assert "cluster_stats" in result
    assert "silhouette_score" in result
    assert "execution_time" in result
    assert "advanced_metrics" in result
    
    assert isinstance(result["kmeans_clusters"], dict)
    assert len(result["kmeans_clusters"]) > 0
    assert all(isinstance(cluster, list) for cluster in result["kmeans_clusters"].values())
    
    assert isinstance(result["cluster_stats"], dict)
    assert len(result["cluster_stats"]) == len(result["kmeans_clusters"])
    assert all(isinstance(stats, dict) for stats in result["cluster_stats"].values())

def test_run_with_insufficient_data(algorithm, api):
    """Test running the algorithm with insufficient data."""
    # Test with empty data
    result = algorithm.run(api, {}, "1d", algorithm.get_default_config())
    assert result == {"kmeans_clusters": {}, "kmeans_centroids": [], "cluster_stats": {}}
    
    # Test with single stock
    single_stock = {"AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0}}
    result = algorithm.run(api, single_stock, "1d", algorithm.get_default_config())
    assert result == {"kmeans_clusters": {}, "kmeans_centroids": [], "cluster_stats": {}}

def test_run_with_custom_config(algorithm, api, sample_data, sample_mesh_results):
    """Test running the algorithm with custom configuration."""
    api.set("mesh_results.1d", sample_mesh_results)
    config = algorithm.get_default_config()
    config.update({
        "n_clusters": 3,
        "clustering_method": ClusteringMethod.DBSCAN,
        "normalization_method": "robust",
        "outlier_removal": True,
        "pca_components": 2
    })
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["kmeans_clusters"]) > 0

def test_result_caching(algorithm, api, sample_data, sample_mesh_results):
    """Test that results are properly cached."""
    api.set("mesh_results.1d", sample_mesh_results)
    algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    cached_result = api.get("algo_cache.kmeans_clustering.1d")
    assert cached_result is not None
    assert "kmeans_clusters" in cached_result
    assert "kmeans_centroids" in cached_result
    assert "cluster_stats" in cached_result
    assert "silhouette_score" in cached_result
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
    large_mesh = {"centrality": {f"STOCK_{i}": 0.5 for i in range(100)}}
    api.set("mesh_results.1d", large_mesh)
    result = algorithm.run(api, large_data, "1d", algorithm.get_default_config())
    assert len(result["kmeans_clusters"]) > 0
    assert result["execution_time"] < 10.0  # Should complete within 10 seconds

def test_error_handling(algorithm, api, sample_data):
    """Test error handling with invalid data."""
    # Test with invalid mesh results
    api.set("mesh_results.1d", {"invalid": "data"})
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    assert result == {"kmeans_clusters": {}, "kmeans_centroids": [], "cluster_stats": {}}

def test_clustering_methods(algorithm, api, sample_data, sample_mesh_results):
    """Test different clustering methods."""
    api.set("mesh_results.1d", sample_mesh_results)
    for method in [ClusteringMethod.KMEANS, ClusteringMethod.DBSCAN, ClusteringMethod.AGGLOMERATIVE]:
        config = algorithm.get_default_config()
        config["clustering_method"] = method
        result = algorithm.run(api, sample_data, "1d", config)
        assert len(result["kmeans_clusters"]) > 0
        if method == ClusteringMethod.KMEANS:
            assert len(result["kmeans_centroids"]) > 0 
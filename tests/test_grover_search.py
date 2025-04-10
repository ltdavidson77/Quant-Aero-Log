import pytest
import numpy as np
import cupy as cp
from algorithms.grover_search import Algorithm, OptimizationLevel, QuantumState
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

def test_algorithm_initialization(algorithm):
    """Test algorithm initialization and configuration."""
    config = algorithm.get_default_config()
    assert isinstance(config, dict)
    assert "search_space" in config
    assert "tolerance" in config
    assert "target_metric" in config
    assert "target_value" in config
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
    assert "quantum_amplitude" in config
    assert "phase_shift" in config
    assert "entanglement_threshold" in config
    assert "market_regime" in config
    assert "time_horizon" in config
    assert "sector_weighting" in config
    assert "liquidity_threshold" in config
    assert "max_iterations" in config
    
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

def test_quantum_state_initialization(algorithm, sample_data):
    """Test initialization of quantum states."""
    corr_matrix = algorithm._compute_correlation_matrix(sample_data)
    states = algorithm._initialize_quantum_states(list(sample_data.keys()), sample_data, corr_matrix)
    
    assert isinstance(states, list)
    assert len(states) == len(sample_data)
    for state in states:
        assert isinstance(state, QuantumState)
        assert state.stock in sample_data
        assert 0 <= state.amplitude <= 1
        assert isinstance(state.phase, float)
        assert isinstance(state.entanglement, list)
        assert all(s in sample_data for s in state.entanglement)

def test_quantum_gate_application(algorithm, sample_data):
    """Test application of quantum gates."""
    advanced_metrics = algorithm._compute_advanced_metrics(sample_data, window=20)
    state = QuantumState("AAPL", 0.5, 0.0, [])
    
    # Test with oracle hit
    new_state = algorithm._apply_quantum_gate(state, True, advanced_metrics)
    assert new_state.amplitude > state.amplitude
    
    # Test with market regime
    config = algorithm.get_default_config()
    config["market_regime"] = "volatile"
    new_state = algorithm._apply_quantum_gate(state, False, advanced_metrics)
    assert new_state.phase != state.phase

def test_diffusion_operator(algorithm):
    """Test application of Grover's diffusion operator."""
    states = [
        QuantumState("A", 0.2, 0.0, []),
        QuantumState("B", 0.3, 0.0, []),
        QuantumState("C", 0.5, 0.0, [])
    ]
    
    new_states = algorithm._apply_diffusion_operator(states)
    assert len(new_states) == len(states)
    assert all(0 <= s.amplitude <= 1 for s in new_states)
    assert abs(sum(s.amplitude ** 2 for s in new_states) - 1.0) < 1e-6

def test_run_with_valid_data(algorithm, api, sample_data):
    """Test running the algorithm with valid data."""
    result = algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    
    assert isinstance(result, dict)
    assert "grover_matches" in result
    assert "grover_confidence" in result
    assert "execution_time" in result
    assert "advanced_metrics" in result
    assert "quantum_states" in result
    
    assert isinstance(result["grover_matches"], list)
    assert len(result["grover_matches"]) > 0
    assert all(isinstance(stock, str) for stock in result["grover_matches"])
    
    assert isinstance(result["grover_confidence"], list)
    assert len(result["grover_confidence"]) == len(result["grover_matches"])
    assert all(isinstance(score, float) for score in result["grover_confidence"])
    
    assert isinstance(result["quantum_states"], list)
    assert len(result["quantum_states"]) > 0
    assert all(isinstance(state, tuple) and len(state) == 4 for state in result["quantum_states"])

def test_run_with_insufficient_data(algorithm, api):
    """Test running the algorithm with insufficient data."""
    # Test with empty data
    result = algorithm.run(api, {}, "1d", algorithm.get_default_config())
    assert result == {"grover_matches": [], "grover_confidence": []}
    
    # Test with single stock
    single_stock = {"AAPL": {"Adj Close": 150.0, "Vol_1d": 1000000, "Volume": 5000000, "High": 155.0, "Low": 145.0, "Open": 148.0, "Close": 150.0}}
    result = algorithm.run(api, single_stock, "1d", algorithm.get_default_config())
    assert result == {"grover_matches": [], "grover_confidence": []}

def test_run_with_custom_config(algorithm, api, sample_data):
    """Test running the algorithm with custom configuration."""
    config = algorithm.get_default_config()
    config.update({
        "target_metric": "Volume",
        "tolerance": 0.1,
        "correlation_method": "spearman",
        "market_regime": "volatile",
        "quantum_amplitude": 1.5,
        "phase_shift": 0.2,
        "max_iterations": 50
    })
    result = algorithm.run(api, sample_data, "1d", config)
    assert len(result["grover_matches"]) > 0

def test_result_caching(algorithm, api, sample_data):
    """Test that results are properly cached."""
    algorithm.run(api, sample_data, "1d", algorithm.get_default_config())
    cached_result = api.get("algo_cache.grover_search.1d")
    assert cached_result is not None
    assert "grover_matches" in cached_result
    assert "grover_confidence" in cached_result
    assert "execution_time" in cached_result
    assert "advanced_metrics" in cached_result
    assert "quantum_states" in cached_result

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
    assert len(result["grover_matches"]) > 0
    assert result["execution_time"] < 10.0  # Should complete within 10 seconds

def test_error_handling(algorithm, api, sample_data):
    """Test error handling with invalid data."""
    # Test with invalid metric
    config = algorithm.get_default_config()
    config["target_metric"] = "InvalidMetric"
    result = algorithm.run(api, sample_data, "1d", config)
    assert result == {"grover_matches": [], "grover_confidence": []}

def test_market_regime_adaptation(algorithm, api, sample_data):
    """Test adaptation to different market regimes."""
    for regime in ["normal", "volatile", "trending"]:
        config = algorithm.get_default_config()
        config["market_regime"] = regime
        result = algorithm.run(api, sample_data, "1d", config)
        assert len(result["grover_matches"]) > 0
        # Verify that the quantum states reflect the market regime
        if regime == "volatile":
            assert any(state[2] != 0.0 for state in result["quantum_states"])  # Check phase shifts
        elif regime == "trending":
            assert any(state[2] != 0.0 for state in result["quantum_states"])  # Check phase shifts 
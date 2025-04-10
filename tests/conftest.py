# ==========================
# tests/conftest.py
# ==========================
# Test configuration and fixtures for the Quant-Aero-Log framework.

import pytest
import numpy as np
from typing import Generator, Dict, Any
from pathlib import Path
import tempfile
import shutil
import logging
from unittest.mock import MagicMock

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tests")

@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create and yield a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_hamiltonian_data() -> Dict[str, Any]:
    """Generate sample data for Hamiltonian tests."""
    return {
        "positions": [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.5, np.sqrt(3)/2])
        ],
        "angles": [0.0, np.pi/6, -np.pi/6],
        "angular_velocities": [0.01, 0.02, 0.015],
        "curvatures": [0.0, 0.0, 0.0],
        "curvature_accelerations": [0.001, 0.001, 0.001]
    }

@pytest.fixture(scope="session")
def mock_logger() -> Generator[MagicMock, None, None]:
    """Create a mock logger for testing logging functionality."""
    mock = MagicMock()
    yield mock

@pytest.fixture(scope="function")
def clean_config() -> Dict[str, Any]:
    """Provide a clean configuration for each test."""
    return {
        "num_pieces": 3,
        "epsilon": 1e-9,
        "learning_rate": 0.01,
        "max_iterations": 1000,
        "tolerance": 1e-6
    }

@pytest.fixture(scope="session")
def performance_benchmark_config() -> Dict[str, Any]:
    """Configuration for performance benchmarks."""
    return {
        "warmup_iterations": 5,
        "measurement_iterations": 10,
        "memory_profiling": True,
        "cpu_profiling": True
    }

@pytest.fixture(scope="function")
def mock_visualization_data() -> Dict[str, Any]:
    """Generate mock data for visualization tests."""
    return {
        "accuracy": np.random.random(100),
        "class_distribution": np.random.randint(0, 10, 100),
        "confusion_matrix": np.random.randint(0, 100, (10, 10)),
        "theta": np.linspace(0, 2*np.pi, 100),
        "phi": np.sin(np.linspace(0, 2*np.pi, 100))
    }

@pytest.fixture(scope="session")
def test_event_bus() -> Generator[MagicMock, None, None]:
    """Create a mock event bus for testing component communication."""
    mock = MagicMock()
    yield mock

@pytest.fixture(scope="function")
def cache_config() -> Dict[str, Any]:
    """Configuration for caching tests."""
    return {
        "max_size": 1000,
        "ttl": 3600,
        "strategy": "lru"
    }

@pytest.fixture(scope="session")
def error_test_cases() -> Dict[str, Any]:
    """Collection of test cases for error handling."""
    return {
        "invalid_input": {
            "positions": [np.array([np.nan, np.nan])],
            "angles": [float('inf')],
            "expected_error": ValueError
        },
        "mismatched_lengths": {
            "positions": [np.array([0, 0])],
            "angles": [0.0, 1.0],
            "expected_error": ValueError
        },
        "invalid_config": {
            "config": {"num_pieces": -1},
            "expected_error": ValueError
        }
    }

@pytest.fixture(scope="function")
def profiling_config() -> Dict[str, Any]:
    """Configuration for performance profiling."""
    return {
        "enable_cpu_profiling": True,
        "enable_memory_profiling": True,
        "enable_gpu_profiling": False,
        "sampling_interval": 0.01
    }

@pytest.fixture(scope="session")
def documentation_config() -> Dict[str, Any]:
    """Configuration for documentation generation."""
    return {
        "output_dir": "docs",
        "api_docs_dir": "api",
        "examples_dir": "examples",
        "tutorials_dir": "tutorials"
    } 
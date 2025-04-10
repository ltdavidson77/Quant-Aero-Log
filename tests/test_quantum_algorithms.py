# ==========================
# tests/test_quantum_algorithms.py
# ==========================
# Test suite for quantum algorithms.

import pytest
import numpy as np
from algorithms.quantum_algorithms import (
    QuantumAlgorithms,
    QuantumAlgorithmConfig
)
from unittest.mock import patch, MagicMock
import time
import gc
import concurrent.futures

@pytest.fixture(scope="session")
def quantum_algo():
    """Create a quantum algorithms instance."""
    config = QuantumAlgorithmConfig(
        num_qubits=4,
        num_layers=3,
        learning_rate=0.01,
        max_iterations=100,
        tolerance=1e-6,
        parallel_processing=True,
        num_threads=4,
        use_gpu=False,
        memory_limit=1024,
        precision="double"
    )
    return QuantumAlgorithms(config)

@pytest.fixture(scope="session")
def sample_hamiltonian():
    """Generate a sample Hamiltonian matrix."""
    n = 16  # 4 qubits
    hamiltonian = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2  # Make Hermitian
    return hamiltonian

@pytest.fixture(scope="session")
def sample_initial_state():
    """Generate a sample initial quantum state."""
    n = 16  # 4 qubits
    state = np.random.rand(n) + 1j * np.random.rand(n)
    return state / np.linalg.norm(state)  # Normalize

@pytest.fixture(scope="session")
def sample_parameters():
    """Generate sample variational parameters."""
    num_qubits = 4
    num_layers = 3
    return np.random.rand(num_qubits * num_layers * 2)  # 2 parameters per qubit per layer

@pytest.fixture(scope="session")
def sample_observable():
    """Generate a sample observable."""
    n = 16  # 4 qubits
    observable = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    observable = (observable + observable.conj().T) / 2  # Make Hermitian
    return observable

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    gc.collect()

def test_initialization(quantum_algo):
    """Test quantum algorithms initialization."""
    assert quantum_algo.config.num_qubits == 4
    assert quantum_algo.config.num_layers == 3
    assert quantum_algo.config.learning_rate == 0.01
    assert quantum_algo.config.max_iterations == 100
    assert quantum_algo.config.tolerance == 1e-6
    assert quantum_algo.config.parallel_processing is True
    assert quantum_algo.config.num_threads == 4
    assert quantum_algo.config.use_gpu is False
    assert quantum_algo.config.memory_limit == 1024
    assert quantum_algo.config.precision == "double"

def test_variational_quantum_eigensolver(quantum_algo, sample_hamiltonian, sample_initial_state, sample_parameters):
    """Test VQE implementation."""
    energy, final_state = quantum_algo.variational_quantum_eigensolver(
        sample_hamiltonian,
        sample_initial_state,
        sample_parameters
    )
    assert isinstance(energy, float)
    assert isinstance(final_state, np.ndarray)
    assert np.allclose(np.linalg.norm(final_state), 1.0)  # State should be normalized

def test_quantum_approximate_optimization(quantum_algo, sample_hamiltonian, sample_parameters):
    """Test QAOA implementation."""
    solution, energy = quantum_algo.quantum_approximate_optimization(
        sample_hamiltonian,
        sample_parameters
    )
    assert isinstance(solution, np.ndarray)
    assert isinstance(energy, float)
    assert len(solution) == quantum_algo.config.num_qubits

def test_quantum_fourier_transform(quantum_algo, sample_initial_state):
    """Test QFT implementation."""
    transformed_state = quantum_algo.quantum_fourier_transform(sample_initial_state)
    assert isinstance(transformed_state, np.ndarray)
    assert np.allclose(np.linalg.norm(transformed_state), 1.0)  # State should be normalized

def test_quantum_phase_estimation(quantum_algo, sample_hamiltonian, sample_initial_state):
    """Test QPE implementation."""
    phases, probabilities = quantum_algo.quantum_phase_estimation(
        sample_hamiltonian,
        sample_initial_state
    )
    assert isinstance(phases, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(phases) == len(probabilities)
    assert np.allclose(np.sum(probabilities), 1.0)  # Probabilities should sum to 1

def test_quantum_gradient_descent(quantum_algo, sample_hamiltonian, sample_initial_state, sample_parameters):
    """Test quantum gradient descent."""
    optimized_params, final_energy = quantum_algo.quantum_gradient_descent(
        sample_hamiltonian,
        sample_initial_state,
        sample_parameters
    )
    assert isinstance(optimized_params, np.ndarray)
    assert isinstance(final_energy, float)
    assert len(optimized_params) == len(sample_parameters)

def test_quantum_entanglement_measurement(quantum_algo, sample_initial_state):
    """Test quantum entanglement measurement."""
    entanglement = quantum_algo.quantum_entanglement_measurement(sample_initial_state)
    assert isinstance(entanglement, float)
    assert 0 <= entanglement <= 1  # Entanglement should be between 0 and 1

def test_quantum_state_tomography(quantum_algo, sample_initial_state):
    """Test quantum state tomography."""
    reconstructed_state = quantum_algo.quantum_state_tomography(sample_initial_state)
    assert isinstance(reconstructed_state, np.ndarray)
    assert np.allclose(np.linalg.norm(reconstructed_state), 1.0)  # State should be normalized

def test_quantum_error_correction(quantum_algo, sample_initial_state):
    """Test quantum error correction."""
    corrected_state = quantum_algo.quantum_error_correction(sample_initial_state)
    assert isinstance(corrected_state, np.ndarray)
    assert np.allclose(np.linalg.norm(corrected_state), 1.0)  # State should be normalized

def test_quantum_measurement(quantum_algo, sample_initial_state, sample_observable):
    """Test quantum measurement."""
    expectation_value = quantum_algo.quantum_measurement(sample_initial_state, sample_observable)
    assert isinstance(expectation_value, float)
    assert np.isreal(expectation_value)  # Expectation value should be real

@pytest.mark.performance
def test_performance_large_hamiltonian(quantum_algo):
    """Test performance with large Hamiltonian."""
    n_qubits = 8
    n = 2**n_qubits
    hamiltonian = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2
    
    start_time = time.time()
    quantum_algo.variational_quantum_eigensolver(
        hamiltonian,
        np.ones(n) / np.sqrt(n),  # Uniform superposition
        np.random.rand(n_qubits * quantum_algo.config.num_layers * 2)
    )
    execution_time = time.time() - start_time
    assert execution_time < 30.0  # Should complete within 30 seconds

def test_error_handling(quantum_algo):
    """Test error handling for invalid inputs."""
    # Test non-Hermitian Hamiltonian
    with pytest.raises(ValueError):
        quantum_algo.variational_quantum_eigensolver(
            np.array([[1, 2], [3, 4]]),  # Non-Hermitian
            np.array([1, 0]),
            np.array([0.5])
        )
    
    # Test non-normalized state
    with pytest.raises(ValueError):
        quantum_algo.quantum_fourier_transform(np.array([1, 1]))  # Not normalized
    
    # Test wrong parameter size
    with pytest.raises(ValueError):
        quantum_algo.quantum_approximate_optimization(
            np.array([[1, 0], [0, 1]]),
            np.array([0.5])  # Wrong size
        )

def test_parallel_processing(quantum_algo, sample_hamiltonian, sample_initial_state):
    """Test parallel processing capabilities."""
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        quantum_algo.variational_quantum_eigensolver(
            sample_hamiltonian,
            sample_initial_state,
            np.random.rand(quantum_algo.config.num_qubits * quantum_algo.config.num_layers * 2)
        )
        mock_executor.assert_called_once()

def test_memory_usage(quantum_algo, sample_hamiltonian):
    """Test memory usage during computation."""
    gc.collect()
    initial_memory = gc.get_objects()
    quantum_algo.variational_quantum_eigensolver(
        sample_hamiltonian,
        np.ones(16) / 4,  # Uniform superposition for 4 qubits
        np.random.rand(quantum_algo.config.num_qubits * quantum_algo.config.num_layers * 2)
    )
    gc.collect()
    final_memory = gc.get_objects()
    assert len(final_memory) - len(initial_memory) < 1000  # Should not create too many new objects

def test_precision_settings(quantum_algo):
    """Test different precision settings."""
    # Test single precision
    quantum_algo.config.precision = "single"
    result = quantum_algo.quantum_fourier_transform(np.array([1, 0]))
    assert result.dtype == np.complex64
    
    # Test double precision
    quantum_algo.config.precision = "double"
    result = quantum_algo.quantum_fourier_transform(np.array([1, 0]))
    assert result.dtype == np.complex128

def test_quantum_circuit_optimization(quantum_algo, sample_parameters):
    """Test quantum circuit optimization."""
    optimized_circuit = quantum_algo.optimize_quantum_circuit(sample_parameters)
    assert isinstance(optimized_circuit, dict)
    assert "gates" in optimized_circuit
    assert "depth" in optimized_circuit
    assert optimized_circuit["depth"] <= len(sample_parameters)  # Depth should not exceed parameter count

def test_quantum_gate_operations(quantum_algo):
    """Test quantum gate operations."""
    # Test Hadamard gate
    state = np.array([1, 0])
    hadamard_state = quantum_algo.apply_hadamard(state)
    assert np.allclose(np.abs(hadamard_state), np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
    
    # Test CNOT gate
    state = np.array([1, 0, 0, 0])  # |00⟩
    cnot_state = quantum_algo.apply_cnot(state)
    assert np.allclose(cnot_state, state)  # Should remain unchanged
    
    # Test Pauli gates
    state = np.array([1, 0])
    x_state = quantum_algo.apply_pauli_x(state)
    assert np.allclose(x_state, np.array([0, 1]))
    
    y_state = quantum_algo.apply_pauli_y(state)
    assert np.allclose(np.abs(y_state), np.array([0, 1]))
    
    z_state = quantum_algo.apply_pauli_z(state)
    assert np.allclose(z_state, state)

def test_quantum_entanglement_generation(quantum_algo):
    """Test quantum entanglement generation."""
    # Test Bell state generation
    state = np.array([1, 0, 0, 0])  # |00⟩
    bell_state = quantum_algo.generate_bell_state(state)
    assert np.allclose(np.abs(bell_state), np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))
    
    # Test GHZ state generation
    state = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # |000⟩
    ghz_state = quantum_algo.generate_ghz_state(state)
    assert np.allclose(np.abs(ghz_state), np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)]))

def test_quantum_algorithm_convergence(quantum_algo, sample_hamiltonian, sample_initial_state, sample_parameters):
    """Test convergence of quantum algorithms."""
    # Test VQE convergence
    energies = []
    for _ in range(10):
        energy, _ = quantum_algo.variational_quantum_eigensolver(
            sample_hamiltonian,
            sample_initial_state,
            sample_parameters
        )
        energies.append(energy)
    assert all(e1 >= e2 for e1, e2 in zip(energies[:-1], energies[1:]))  # Energy should decrease
    
    # Test QAOA convergence
    energies = []
    for _ in range(10):
        _, energy = quantum_algo.quantum_approximate_optimization(
            sample_hamiltonian,
            sample_parameters
        )
        energies.append(energy)
    assert all(e1 >= e2 for e1, e2 in zip(energies[:-1], energies[1:]))  # Energy should decrease 
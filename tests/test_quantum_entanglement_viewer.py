# ==========================
# tests/test_quantum_entanglement_viewer.py
# ==========================
# Test suite for quantum entanglement visualization.

import pytest
import numpy as np
from visuals.quantum_entanglement_viewer import (
    QuantumEntanglementViewer,
    EntanglementVizConfig
)
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

@pytest.fixture
def viewer():
    """Create a quantum entanglement viewer instance."""
    config = EntanglementVizConfig(
        figsize=(8, 6),
        dpi=100,
        fps=30,
        colormap="viridis",
        alpha=0.7,
        linewidth=2.0,
        point_size=50.0,
        show_entanglement_lines=True,
        show_density_matrix=True,
        show_correlation=True
    )
    return QuantumEntanglementViewer(config)

@pytest.fixture
def sample_density_matrix():
    """Generate a sample density matrix."""
    n_states = 5
    density_matrix = np.random.rand(n_states, n_states) + 1j * np.random.rand(n_states, n_states)
    density_matrix = (density_matrix + density_matrix.conj().T) / 2  # Make Hermitian
    density_matrix = density_matrix / np.trace(density_matrix)  # Normalize
    return density_matrix

@pytest.fixture
def sample_density_sequence(sample_density_matrix):
    """Generate a sequence of density matrices for animation."""
    n_frames = 10
    sequence = []
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        density_dyn = sample_density_matrix * np.exp(1j * phase)
        density_dyn = (density_dyn + density_dyn.conj().T) / 2
        density_dyn = density_dyn / np.trace(density_dyn)
        sequence.append(density_dyn)
    return sequence

def test_viewer_initialization(viewer):
    """Test viewer initialization."""
    assert viewer.config.figsize == (8, 6)
    assert viewer.config.dpi == 100
    assert viewer.config.fps == 30
    assert viewer.config.colormap == "viridis"
    assert viewer.config.alpha == 0.7
    assert viewer.config.linewidth == 2.0
    assert viewer.config.point_size == 50.0
    assert viewer.config.show_entanglement_lines is True
    assert viewer.config.show_density_matrix is True
    assert viewer.config.show_correlation is True

def test_plot_entanglement_network(viewer, sample_density_matrix, tmp_path):
    """Test entanglement network plotting."""
    save_path = tmp_path / "network.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_entanglement_network(sample_density_matrix, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_create_entanglement_animation(viewer, sample_density_sequence, tmp_path):
    """Test entanglement animation creation."""
    viewer.config.save_animation = True
    viewer.config.output_path = str(tmp_path / "animation.mp4")
    with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
        anim = viewer.create_entanglement_animation(sample_density_sequence)
        assert anim is not None
        mock_save.assert_called_once()

def test_plot_density_matrix(viewer, sample_density_matrix, tmp_path):
    """Test density matrix plotting."""
    save_path = tmp_path / "density.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_density_matrix(sample_density_matrix, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_plot_correlation_matrix(viewer, sample_density_matrix, tmp_path):
    """Test correlation matrix plotting."""
    save_path = tmp_path / "correlation.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_correlation_matrix(sample_density_matrix, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_plot_entanglement_spectrum(viewer, sample_density_matrix, tmp_path):
    """Test entanglement spectrum plotting."""
    save_path = tmp_path / "spectrum.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_entanglement_spectrum(sample_density_matrix, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_invalid_density_matrix(viewer):
    """Test handling of invalid density matrix."""
    invalid_matrix = np.array([[1, 0], [0, 1]])  # Not normalized
    with pytest.raises(ValueError):
        viewer.plot_entanglement_network(invalid_matrix)

def test_empty_density_sequence(viewer):
    """Test handling of empty density sequence."""
    with pytest.raises(ValueError):
        viewer.create_entanglement_animation([])

def test_style_setup(viewer):
    """Test style setup."""
    with patch('matplotlib.pyplot.style.use') as mock_style:
        with patch('seaborn.set_palette') as mock_palette:
            viewer._setup_style()
            mock_style.assert_called_once_with('seaborn')
            mock_palette.assert_called_once_with(viewer.config.colormap)

def test_custom_positions(viewer, sample_density_matrix):
    """Test plotting with custom positions."""
    positions = np.random.rand(sample_density_matrix.shape[0], 3)
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_entanglement_network(sample_density_matrix, positions=positions)
        mock_show.assert_called_once()

def test_animation_with_positions(viewer, sample_density_sequence):
    """Test animation with custom positions."""
    positions = np.random.rand(sample_density_sequence[0].shape[0], 3)
    with patch('matplotlib.animation.FuncAnimation') as mock_anim:
        viewer.create_entanglement_animation(sample_density_sequence, positions=positions)
        mock_anim.assert_called_once()

def test_entanglement_threshold(viewer, sample_density_matrix):
    """Test entanglement threshold behavior."""
    # Create a matrix with known entanglement values
    test_matrix = np.array([
        [0.5, 0.2, 0.1],
        [0.2, 0.3, 0.05],
        [0.1, 0.05, 0.2]
    ])
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_entanglement_network(test_matrix)
        mock_show.assert_called_once()

def test_visualization_consistency(viewer, sample_density_matrix):
    """Test consistency of visualizations."""
    # Test that the same input produces consistent output
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_density_matrix(sample_density_matrix)
        viewer.plot_correlation_matrix(sample_density_matrix)
        viewer.plot_entanglement_spectrum(sample_density_matrix)
        assert mock_show.call_count == 3

def test_performance_large_matrices(viewer):
    """Test performance with large matrices."""
    # Create a large density matrix
    n_states = 100
    large_matrix = np.random.rand(n_states, n_states) + 1j * np.random.rand(n_states, n_states)
    large_matrix = (large_matrix + large_matrix.conj().T) / 2
    large_matrix = large_matrix / np.trace(large_matrix)
    
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_density_matrix(large_matrix)
        mock_show.assert_called_once()

def test_error_handling(viewer):
    """Test error handling for invalid inputs."""
    # Test non-square matrix
    with pytest.raises(ValueError):
        viewer.plot_density_matrix(np.array([[1, 0], [0, 1], [0, 0]]))
    
    # Test non-Hermitian matrix
    with pytest.raises(ValueError):
        viewer.plot_density_matrix(np.array([[1, 2], [3, 4]]))
    
    # Test non-positive definite matrix
    with pytest.raises(ValueError):
        viewer.plot_density_matrix(np.array([[-1, 0], [0, -1]])) 
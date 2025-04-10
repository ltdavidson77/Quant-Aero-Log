# ==========================
# tests/test_hamiltonian_energy_map.py
# ==========================
# Test suite for Hamiltonian energy map visualization.

import pytest
import numpy as np
from visuals.hamiltonian_energy_map import (
    HamiltonianEnergyMap,
    EnergyMapConfig
)
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

@pytest.fixture
def visualizer():
    """Create a Hamiltonian energy map visualizer instance."""
    config = EnergyMapConfig(
        figsize=(8, 6),
        dpi=100,
        fps=30,
        colormap="viridis",
        alpha=0.7,
        contour_levels=20,
        show_contours=True,
        show_surface=True,
        show_minima=True,
        show_trajectory=True
    )
    return HamiltonianEnergyMap(config)

@pytest.fixture
def sample_energy_data():
    """Generate sample energy data."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    energy = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)
    return X, Y, energy

@pytest.fixture
def sample_trajectory():
    """Generate sample trajectory data."""
    t = np.linspace(0, 10, 100)
    return np.column_stack((np.sin(t), np.cos(t)))

@pytest.fixture
def sample_minima():
    """Generate sample minima coordinates."""
    return [(0, 0), (np.pi, np.pi), (-np.pi, -np.pi)]

@pytest.fixture
def sample_energy_sequence(sample_energy_data):
    """Generate a sequence of energy surfaces for animation."""
    X, Y, base_energy = sample_energy_data
    n_frames = 10
    sequence = []
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        energy_dyn = np.sin(X + phase) * np.cos(Y + phase) + 0.1 * (X**2 + Y**2)
        sequence.append(energy_dyn)
    return sequence

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer.config.figsize == (8, 6)
    assert visualizer.config.dpi == 100
    assert visualizer.config.fps == 30
    assert visualizer.config.colormap == "viridis"
    assert visualizer.config.alpha == 0.7
    assert visualizer.config.contour_levels == 20
    assert visualizer.config.show_contours is True
    assert visualizer.config.show_surface is True
    assert visualizer.config.show_minima is True
    assert visualizer.config.show_trajectory is True

def test_plot_energy_surface(visualizer, sample_energy_data, sample_trajectory, sample_minima, tmp_path):
    """Test energy surface plotting."""
    X, Y, energy = sample_energy_data
    save_path = tmp_path / "surface.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_surface(X, Y, energy, sample_trajectory, sample_minima, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_create_energy_animation(visualizer, sample_energy_data, sample_energy_sequence, tmp_path):
    """Test energy animation creation."""
    X, Y, _ = sample_energy_data
    visualizer.config.save_animation = True
    visualizer.config.output_path = str(tmp_path / "animation.mp4")
    with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
        anim = visualizer.create_energy_animation(X, Y, sample_energy_sequence)
        assert anim is not None
        mock_save.assert_called_once()

def test_plot_energy_contours(visualizer, sample_energy_data, sample_trajectory, sample_minima, tmp_path):
    """Test energy contour plotting."""
    X, Y, energy = sample_energy_data
    save_path = tmp_path / "contours.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_contours(X, Y, energy, sample_trajectory, sample_minima, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_plot_energy_histogram(visualizer, sample_energy_data, tmp_path):
    """Test energy histogram plotting."""
    _, _, energy = sample_energy_data
    save_path = tmp_path / "histogram.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_histogram(energy.flatten(), save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_invalid_input_shapes(visualizer):
    """Test handling of invalid input shapes."""
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 50)  # Different length
    energy = np.random.rand(100, 50)
    with pytest.raises(ValueError):
        visualizer.plot_energy_surface(X, Y, energy)

def test_empty_energy_sequence(visualizer, sample_energy_data):
    """Test handling of empty energy sequence."""
    X, Y, _ = sample_energy_data
    with pytest.raises(ValueError):
        visualizer.create_energy_animation(X, Y, [])

def test_style_setup(visualizer):
    """Test style setup."""
    with patch('matplotlib.pyplot.style.use') as mock_style:
        with patch('seaborn.set_palette') as mock_palette:
            visualizer._setup_style()
            mock_style.assert_called_once_with('seaborn')
            mock_palette.assert_called_once_with(visualizer.config.colormap)

def test_visualization_consistency(visualizer, sample_energy_data, sample_trajectory, sample_minima):
    """Test consistency of visualizations."""
    X, Y, energy = sample_energy_data
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_surface(X, Y, energy, sample_trajectory, sample_minima)
        visualizer.plot_energy_contours(X, Y, energy, sample_trajectory, sample_minima)
        visualizer.plot_energy_histogram(energy.flatten())
        assert mock_show.call_count == 3

def test_performance_large_data(visualizer):
    """Test performance with large datasets."""
    # Create large energy surface
    x = np.linspace(-5, 5, 1000)
    y = np.linspace(-5, 5, 1000)
    X, Y = np.meshgrid(x, y)
    energy = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)
    
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_surface(X, Y, energy)
        mock_show.assert_called_once()

def test_error_handling(visualizer):
    """Test error handling for invalid inputs."""
    # Test invalid trajectory shape
    with pytest.raises(ValueError):
        visualizer.plot_energy_surface(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([[0, 1], [1, 0]]),
            trajectory=np.array([0, 1, 2])  # Wrong shape
        )
    
    # Test invalid minima format
    with pytest.raises(ValueError):
        visualizer.plot_energy_surface(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([[0, 1], [1, 0]]),
            minima=[(0,)]  # Wrong format
        )
    
    # Test non-numeric input
    with pytest.raises(ValueError):
        visualizer.plot_energy_surface(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([['a', 'b'], ['c', 'd']])  # Non-numeric
        )

def test_animation_with_trajectory(visualizer, sample_energy_data, sample_energy_sequence, sample_trajectory):
    """Test animation with trajectory."""
    X, Y, _ = sample_energy_data
    with patch('matplotlib.animation.FuncAnimation') as mock_anim:
        visualizer.create_energy_animation(X, Y, sample_energy_sequence, sample_trajectory)
        mock_anim.assert_called_once()

def test_contour_levels(visualizer, sample_energy_data):
    """Test contour level configuration."""
    X, Y, energy = sample_energy_data
    visualizer.config.contour_levels = 5
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_energy_contours(X, Y, energy)
        mock_show.assert_called_once() 
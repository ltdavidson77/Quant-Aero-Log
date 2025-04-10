# ==========================
# tests/test_quantum_state_visualizer.py
# ==========================
# Test suite for quantum state visualization.

import pytest
import numpy as np
from visuals.quantum_state_visualizer import (
    QuantumStateVisualizer,
    QuantumVizConfig
)
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

@pytest.fixture
def visualizer():
    """Create a quantum state visualizer instance."""
    config = QuantumVizConfig(
        figsize=(8, 6),
        dpi=100,
        fps=30,
        trail_length=50,
        colormap="viridis",
        alpha=0.7,
        linewidth=2.0,
        point_size=50.0,
        show_entanglement=True,
        show_probability=True,
        show_phase=True,
        save_animation=False
    )
    return QuantumStateVisualizer(config)

@pytest.fixture
def sample_quantum_states():
    """Generate sample quantum states."""
    n_states = 5
    n_points = 100
    states = []
    for _ in range(n_points):
        state = np.random.rand(n_states) + 1j * np.random.rand(n_states)
        state = state / np.linalg.norm(state)  # Normalize
        states.append(state)
    return states

@pytest.fixture
def sample_times():
    """Generate sample time points."""
    return np.linspace(0, 10, 100)

@pytest.fixture
def sample_entanglement():
    """Generate sample entanglement measures."""
    return np.random.rand(100)

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization."""
    assert visualizer.config.figsize == (8, 6)
    assert visualizer.config.dpi == 100
    assert visualizer.config.fps == 30
    assert visualizer.config.trail_length == 50
    assert visualizer.config.colormap == "viridis"
    assert visualizer.config.alpha == 0.7
    assert visualizer.config.linewidth == 2.0
    assert visualizer.config.point_size == 50.0
    assert visualizer.config.show_entanglement is True
    assert visualizer.config.show_probability is True
    assert visualizer.config.show_phase is True

def test_plot_state_evolution(visualizer, sample_quantum_states, sample_times, sample_entanglement, tmp_path):
    """Test state evolution plotting."""
    save_path = tmp_path / "evolution.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(sample_quantum_states, sample_times, sample_entanglement, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_create_3d_animation(visualizer, sample_quantum_states, sample_times, sample_entanglement, tmp_path):
    """Test 3D animation creation."""
    visualizer.config.save_animation = True
    visualizer.config.output_path = str(tmp_path / "animation.mp4")
    with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
        anim = visualizer.create_3d_animation(sample_quantum_states, sample_times, sample_entanglement)
        assert anim is not None
        mock_save.assert_called_once()

def test_plot_phase_space(visualizer, sample_quantum_states, sample_times, tmp_path):
    """Test phase space plotting."""
    save_path = tmp_path / "phase_space.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_phase_space(sample_quantum_states, sample_times, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_plot_entanglement_matrix(visualizer, sample_quantum_states, tmp_path):
    """Test entanglement matrix plotting."""
    save_path = tmp_path / "entanglement.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_entanglement_matrix(sample_quantum_states, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_invalid_states(visualizer):
    """Test handling of invalid quantum states."""
    # Test non-normalized states
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution([np.array([1, 1])], np.array([0]))
    
    # Test states with different dimensions
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution([
            np.array([1, 0]),
            np.array([1, 0, 0])
        ], np.array([0, 1]))

def test_empty_states(visualizer):
    """Test handling of empty state list."""
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution([], np.array([]))

def test_style_setup(visualizer):
    """Test style setup."""
    with patch('matplotlib.pyplot.style.use') as mock_style:
        with patch('seaborn.set_palette') as mock_palette:
            visualizer._setup_style()
            mock_style.assert_called_once_with('seaborn')
            mock_palette.assert_called_once_with(visualizer.config.colormap)

def test_visualization_consistency(visualizer, sample_quantum_states, sample_times, sample_entanglement):
    """Test consistency of visualizations."""
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(sample_quantum_states, sample_times, sample_entanglement)
        visualizer.plot_phase_space(sample_quantum_states, sample_times)
        visualizer.plot_entanglement_matrix(sample_quantum_states)
        assert mock_show.call_count == 3

def test_performance_large_states(visualizer):
    """Test performance with large state vectors."""
    # Create large quantum states
    n_states = 100
    n_points = 1000
    states = []
    for _ in range(n_points):
        state = np.random.rand(n_states) + 1j * np.random.rand(n_states)
        state = state / np.linalg.norm(state)
        states.append(state)
    
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(states, np.linspace(0, 10, n_points))
        mock_show.assert_called_once()

def test_error_handling(visualizer):
    """Test error handling for invalid inputs."""
    # Test invalid time array length
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution(
            [np.array([1, 0]), np.array([0, 1])],
            np.array([0])  # Wrong length
        )
    
    # Test invalid entanglement array length
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution(
            [np.array([1, 0]), np.array([0, 1])],
            np.array([0, 1]),
            entanglement=np.array([0.5])  # Wrong length
        )
    
    # Test non-complex states
    with pytest.raises(ValueError):
        visualizer.plot_state_evolution(
            [np.array([1, 0])],  # Not complex
            np.array([0])
        )

def test_animation_with_entanglement(visualizer, sample_quantum_states, sample_times, sample_entanglement):
    """Test animation with entanglement."""
    with patch('matplotlib.animation.FuncAnimation') as mock_anim:
        visualizer.create_3d_animation(sample_quantum_states, sample_times, sample_entanglement)
        mock_anim.assert_called_once()

def test_visualization_options(visualizer, sample_quantum_states, sample_times):
    """Test different visualization options."""
    # Test without entanglement
    visualizer.config.show_entanglement = False
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(sample_quantum_states, sample_times)
        mock_show.assert_called_once()
    
    # Test without probability
    visualizer.config.show_probability = False
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(sample_quantum_states, sample_times)
        mock_show.assert_called_once()
    
    # Test without phase
    visualizer.config.show_phase = False
    with patch('matplotlib.pyplot.show') as mock_show:
        visualizer.plot_state_evolution(sample_quantum_states, sample_times)
        mock_show.assert_called_once() 
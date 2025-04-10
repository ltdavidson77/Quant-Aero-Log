# ==========================
# tests/test_lattice_topology_viewer.py
# ==========================
# Test suite for lattice topology visualization.

import pytest
import numpy as np
import networkx as nx
from visuals.lattice_topology_viewer import (
    LatticeTopologyViewer,
    LatticeVizConfig
)
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import time
import gc

@pytest.fixture(scope="session")
def viewer():
    """Create a lattice topology viewer instance."""
    config = LatticeVizConfig(
        figsize=(8, 6),
        dpi=100,
        fps=30,
        node_size=300,
        edge_width=2.0,
        colormap="viridis",
        alpha=0.7,
        show_labels=True,
        show_weights=True,
        layout="spring",
        save_animation=False
    )
    return LatticeTopologyViewer(config)

@pytest.fixture(scope="session")
def sample_adjacency():
    """Generate a sample adjacency matrix."""
    n_nodes = 5
    adjacency = np.random.rand(n_nodes, n_nodes)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency, 0)  # Zero diagonal
    return adjacency

@pytest.fixture(scope="session")
def sample_node_positions():
    """Generate sample node positions."""
    n_nodes = 5
    return np.random.rand(n_nodes, 3)

@pytest.fixture(scope="session")
def sample_node_colors():
    """Generate sample node colors."""
    n_nodes = 5
    return np.random.rand(n_nodes)

@pytest.fixture(scope="session")
def sample_edge_weights(sample_adjacency):
    """Generate sample edge weights."""
    return sample_adjacency

@pytest.fixture(scope="session")
def sample_adjacency_sequence(sample_adjacency):
    """Generate a sequence of adjacency matrices for animation."""
    n_frames = 10
    sequence = []
    for i in range(n_frames):
        phase = 2 * np.pi * i / n_frames
        adjacency_dyn = sample_adjacency * np.cos(phase)
        adjacency_dyn = (adjacency_dyn + adjacency_dyn.T) / 2
        np.fill_diagonal(adjacency_dyn, 0)
        sequence.append(adjacency_dyn)
    return sequence

@pytest.fixture(scope="session")
def large_adjacency():
    """Generate a large adjacency matrix for performance testing."""
    n_nodes = 1000
    adjacency = np.random.rand(n_nodes, n_nodes)
    adjacency = (adjacency + adjacency.T) / 2
    np.fill_diagonal(adjacency, 0)
    return adjacency

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    plt.close('all')
    gc.collect()

def test_viewer_initialization(viewer):
    """Test viewer initialization."""
    assert viewer.config.figsize == (8, 6)
    assert viewer.config.dpi == 100
    assert viewer.config.fps == 30
    assert viewer.config.node_size == 300
    assert viewer.config.edge_width == 2.0
    assert viewer.config.colormap == "viridis"
    assert viewer.config.alpha == 0.7
    assert viewer.config.show_labels is True
    assert viewer.config.show_weights is True
    assert viewer.config.layout == "spring"

def test_plot_lattice(viewer, sample_adjacency, sample_node_positions, sample_node_colors, sample_edge_weights, tmp_path):
    """Test lattice plotting."""
    save_path = tmp_path / "lattice.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(
            sample_adjacency,
            positions=sample_node_positions,
            node_colors=sample_node_colors,
            edge_weights=sample_edge_weights,
            save_path=str(save_path)
        )
        mock_show.assert_called_once()
    assert save_path.exists()

def test_create_3d_animation(viewer, sample_adjacency_sequence, sample_node_positions, tmp_path):
    """Test 3D animation creation."""
    viewer.config.save_animation = True
    viewer.config.output_path = str(tmp_path / "animation.mp4")
    with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
        anim = viewer.create_3d_animation(sample_adjacency_sequence, positions=sample_node_positions)
        assert anim is not None
        mock_save.assert_called_once()

def test_plot_energy_landscape(viewer, sample_adjacency, sample_node_colors, tmp_path):
    """Test energy landscape plotting."""
    save_path = tmp_path / "energy.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_energy_landscape(sample_adjacency, sample_node_colors, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_plot_connectivity_matrix(viewer, sample_adjacency, tmp_path):
    """Test connectivity matrix plotting."""
    save_path = tmp_path / "connectivity.png"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_connectivity_matrix(sample_adjacency, save_path=str(save_path))
        mock_show.assert_called_once()
    assert save_path.exists()

def test_invalid_adjacency(viewer):
    """Test handling of invalid adjacency matrix."""
    # Test non-square matrix
    with pytest.raises(ValueError):
        viewer.plot_lattice(np.array([[1, 0], [0, 1], [0, 0]]))
    
    # Test non-symmetric matrix
    with pytest.raises(ValueError):
        viewer.plot_lattice(np.array([[0, 1], [2, 0]]))
    
    # Test matrix with non-zero diagonal
    with pytest.raises(ValueError):
        viewer.plot_lattice(np.array([[1, 0], [0, 1]]))

def test_empty_adjacency_sequence(viewer):
    """Test handling of empty adjacency sequence."""
    with pytest.raises(ValueError):
        viewer.create_3d_animation([])

def test_style_setup(viewer):
    """Test style setup."""
    with patch('matplotlib.pyplot.style.use') as mock_style:
        with patch('seaborn.set_palette') as mock_palette:
            viewer._setup_style()
            mock_style.assert_called_once_with('seaborn')
            mock_palette.assert_called_once_with(viewer.config.colormap)

def test_visualization_consistency(viewer, sample_adjacency, sample_node_colors):
    """Test consistency of visualizations."""
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency)
        viewer.plot_energy_landscape(sample_adjacency, sample_node_colors)
        viewer.plot_connectivity_matrix(sample_adjacency)
        assert mock_show.call_count == 3

@pytest.mark.performance
def test_performance_large_lattice(viewer, large_adjacency):
    """Test performance with large adjacency matrix."""
    start_time = time.time()
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(large_adjacency)
        mock_show.assert_called_once()
    execution_time = time.time() - start_time
    assert execution_time < 5.0  # Should complete within 5 seconds

def test_error_handling(viewer):
    """Test error handling for invalid inputs."""
    # Test invalid positions shape
    with pytest.raises(ValueError):
        viewer.plot_lattice(
            np.array([[0, 1], [1, 0]]),
            positions=np.array([[0, 0]])  # Wrong shape
        )
    
    # Test invalid node colors length
    with pytest.raises(ValueError):
        viewer.plot_lattice(
            np.array([[0, 1], [1, 0]]),
            node_colors=np.array([0.5])  # Wrong length
        )
    
    # Test invalid edge weights shape
    with pytest.raises(ValueError):
        viewer.plot_lattice(
            np.array([[0, 1], [1, 0]]),
            edge_weights=np.array([[0, 1], [1, 0], [0, 0]])  # Wrong shape
        )

def test_animation_with_positions(viewer, sample_adjacency_sequence, sample_node_positions):
    """Test animation with node positions."""
    with patch('matplotlib.animation.FuncAnimation') as mock_anim:
        viewer.create_3d_animation(sample_adjacency_sequence, positions=sample_node_positions)
        mock_anim.assert_called_once()

def test_layout_options(viewer, sample_adjacency):
    """Test different layout options."""
    # Test spring layout
    viewer.config.layout = "spring"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency)
        mock_show.assert_called_once()
    
    # Test circular layout
    viewer.config.layout = "circular"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency)
        mock_show.assert_called_once()
    
    # Test random layout
    viewer.config.layout = "random"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency)
        mock_show.assert_called_once()

def test_node_and_edge_attributes(viewer, sample_adjacency, sample_node_colors, sample_edge_weights):
    """Test visualization with node and edge attributes."""
    # Test with node colors
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency, node_colors=sample_node_colors)
        mock_show.assert_called_once()
    
    # Test with edge weights
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency, edge_weights=sample_edge_weights)
        mock_show.assert_called_once()
    
    # Test with both
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(
            sample_adjacency,
            node_colors=sample_node_colors,
            edge_weights=sample_edge_weights
        )
        mock_show.assert_called_once()

@pytest.mark.performance
def test_memory_usage(viewer, large_adjacency):
    """Test memory usage during visualization."""
    gc.collect()
    initial_memory = gc.get_objects()
    with patch('matplotlib.pyplot.show'):
        viewer.plot_lattice(large_adjacency)
    gc.collect()
    final_memory = gc.get_objects()
    assert len(final_memory) - len(initial_memory) < 1000  # Should not create too many new objects

def test_parallel_processing(viewer, sample_adjacency_sequence):
    """Test parallel processing capabilities."""
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        viewer.create_3d_animation(sample_adjacency_sequence)
        mock_executor.assert_called_once()

def test_custom_colormap(viewer, sample_adjacency):
    """Test custom colormap support."""
    viewer.config.colormap = "plasma"
    with patch('matplotlib.pyplot.show') as mock_show:
        viewer.plot_lattice(sample_adjacency)
        mock_show.assert_called_once()

def test_animation_fps(viewer, sample_adjacency_sequence):
    """Test animation FPS configuration."""
    viewer.config.fps = 60
    with patch('matplotlib.animation.FuncAnimation') as mock_anim:
        viewer.create_3d_animation(sample_adjacency_sequence)
        mock_anim.assert_called_once()
        assert mock_anim.call_args[1]['interval'] == 1000 // 60  # Check FPS conversion to interval 
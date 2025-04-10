# ==========================
# visuals/lattice_topology_viewer.py
# ==========================
# Advanced visualization of lattice topologies and their evolution.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx
import seaborn as sns
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("visuals.lattice")

@dataclass
class LatticeVizConfig:
    """Configuration for lattice topology visualization."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    fps: int = 30
    node_size: float = 500.0
    edge_width: float = 2.0
    colormap: str = "viridis"
    alpha: float = 0.7
    show_labels: bool = True
    show_weights: bool = True
    layout: str = "spring"  # spring, spectral, or random
    save_animation: bool = False
    output_path: str = "lattice_topology_animation.mp4"

class LatticeTopologyViewer:
    """
    Advanced visualization of lattice topologies with dynamic updates.
    """
    
    def __init__(self, config: Optional[LatticeVizConfig] = None):
        """
        Initialize the lattice topology viewer.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = LatticeVizConfig()
        
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup visualization style."""
        plt.style.use('seaborn')
        sns.set_palette(self.config.colormap)
    
    def plot_lattice(self,
                    adjacency: np.ndarray,
                    node_positions: Optional[np.ndarray] = None,
                    node_colors: Optional[np.ndarray] = None,
                    edge_weights: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None) -> None:
        """
        Plot a static lattice topology.
        
        Args:
            adjacency: Adjacency matrix
            node_positions: Optional node positions
            node_colors: Optional node colors
            edge_weights: Optional edge weights
            save_path: Optional path to save the figure
        """
        # Create graph
        G = nx.from_numpy_array(adjacency)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Compute layout if not provided
        if node_positions is None:
            if self.config.layout == "spring":
                pos = nx.spring_layout(G)
            elif self.config.layout == "spectral":
                pos = nx.spectral_layout(G)
            else:
                pos = nx.random_layout(G)
        else:
            pos = {i: node_positions[i] for i in range(len(node_positions))}
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             node_size=self.config.node_size,
                             node_color=node_colors,
                             alpha=self.config.alpha,
                             ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                             width=self.config.edge_width,
                             alpha=self.config.alpha,
                             ax=ax)
        
        # Add labels
        if self.config.show_labels:
            nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Add edge weights
        if self.config.show_weights and edge_weights is not None:
            edge_labels = {(i, j): f"{edge_weights[i, j]:.2f}"
                         for i, j in G.edges()}
            nx.draw_networkx_edge_labels(G, pos,
                                       edge_labels=edge_labels,
                                       ax=ax)
        
        ax.set_title("Lattice Topology")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("lattice_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def create_3d_animation(self,
                          adjacency_sequence: List[np.ndarray],
                          node_positions: Optional[List[np.ndarray]] = None,
                          node_colors: Optional[List[np.ndarray]] = None,
                          edge_weights: Optional[List[np.ndarray]] = None) -> FuncAnimation:
        """
        Create a 3D animation of lattice topology evolution.
        
        Args:
            adjacency_sequence: Sequence of adjacency matrices
            node_positions: Optional sequence of node positions
            node_colors: Optional sequence of node colors
            edge_weights: Optional sequence of edge weights
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Get current state
            adj = adjacency_sequence[frame]
            pos = (node_positions[frame] if node_positions is not None
                  else None)
            colors = (node_colors[frame] if node_colors is not None
                     else None)
            weights = (edge_weights[frame] if edge_weights is not None
                      else None)
            
            # Create graph
            G = nx.from_numpy_array(adj)
            
            # Compute 3D positions if not provided
            if pos is None:
                pos = nx.spring_layout(G, dim=3)
            else:
                pos = {i: pos[i] for i in range(len(pos))}
            
            # Extract coordinates
            x = [pos[i][0] for i in range(len(pos))]
            y = [pos[i][1] for i in range(len(pos))]
            z = [pos[i][2] for i in range(len(pos))]
            
            # Plot nodes
            scatter = ax.scatter(x, y, z,
                               c=colors,
                               s=self.config.node_size,
                               alpha=self.config.alpha)
            
            # Plot edges
            for edge in G.edges():
                i, j = edge
                ax.plot([pos[i][0], pos[j][0]],
                       [pos[i][1], pos[j][1]],
                       [pos[i][2], pos[j][2]],
                       'k-',
                       alpha=self.config.alpha,
                       linewidth=self.config.edge_width)
            
            ax.set_title(f"Frame {frame}")
            return [scatter]
        
        anim = FuncAnimation(fig, update,
                           frames=len(adjacency_sequence),
                           interval=1000/self.config.fps,
                           blit=True)
        
        if self.config.save_animation:
            anim.save(self.config.output_path,
                     writer='ffmpeg',
                     fps=self.config.fps,
                     dpi=self.config.dpi)
            logger.info("lattice_animation_saved",
                       path=self.config.output_path)
        
        return anim
    
    def plot_energy_landscape(self,
                            adjacency: np.ndarray,
                            energies: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """
        Plot the energy landscape of the lattice.
        
        Args:
            adjacency: Adjacency matrix
            energies: Node energies
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Create graph
        G = nx.from_numpy_array(adjacency)
        
        # Compute layout
        pos = nx.spring_layout(G)
        
        # Draw nodes with energy-based colors
        nx.draw_networkx_nodes(G, pos,
                             node_size=self.config.node_size,
                             node_color=energies,
                             cmap=self.config.colormap,
                             alpha=self.config.alpha,
                             ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                             width=self.config.edge_width,
                             alpha=self.config.alpha,
                             ax=ax)
        
        # Add energy labels
        if self.config.show_labels:
            labels = {i: f"{energies[i]:.2f}" for i in range(len(energies))}
            nx.draw_networkx_labels(G, pos, labels, ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.config.colormap)
        sm.set_array(energies)
        plt.colorbar(sm, ax=ax, label="Energy")
        
        ax.set_title("Energy Landscape")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("energy_landscape_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def plot_connectivity_matrix(self,
                               adjacency: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        Plot the connectivity matrix of the lattice.
        
        Args:
            adjacency: Adjacency matrix
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot heatmap
        sns.heatmap(adjacency,
                   cmap=self.config.colormap,
                   annot=self.config.show_weights,
                   fmt=".2f",
                   ax=ax)
        
        ax.set_title("Connectivity Matrix")
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("connectivity_matrix_plot_saved",
                       path=save_path)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create viewer
    config = LatticeVizConfig(
        figsize=(10, 8),
        colormap="plasma",
        show_labels=True,
        show_weights=True
    )
    viewer = LatticeTopologyViewer(config)
    
    # Generate sample data
    n_nodes = 10
    adjacency = np.random.rand(n_nodes, n_nodes)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency, 0)  # Zero diagonal
    
    node_positions = np.random.rand(n_nodes, 3)
    node_colors = np.random.rand(n_nodes)
    edge_weights = adjacency.copy()
    
    # Create sequence for animation
    n_frames = 50
    adjacency_sequence = []
    positions_sequence = []
    colors_sequence = []
    weights_sequence = []
    
    for i in range(n_frames):
        # Add some dynamics
        adj = adjacency + 0.1 * np.random.randn(n_nodes, n_nodes)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        adjacency_sequence.append(adj)
        
        pos = node_positions + 0.1 * np.random.randn(n_nodes, 3)
        positions_sequence.append(pos)
        
        colors = node_colors + 0.1 * np.random.randn(n_nodes)
        colors_sequence.append(colors)
        
        weights = edge_weights + 0.1 * np.random.randn(n_nodes, n_nodes)
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        weights_sequence.append(weights)
    
    # Create visualizations
    viewer.plot_lattice(adjacency, node_positions, node_colors, edge_weights)
    anim = viewer.create_3d_animation(adjacency_sequence, positions_sequence,
                                    colors_sequence, weights_sequence)
    viewer.plot_energy_landscape(adjacency, node_colors)
    viewer.plot_connectivity_matrix(adjacency) 
# ==========================
# visuals/quantum_entanglement_viewer.py
# ==========================
# Advanced visualization of quantum entanglement.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("visuals.entanglement")

@dataclass
class EntanglementVizConfig:
    """Configuration for quantum entanglement visualization."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    fps: int = 30
    colormap: str = "viridis"
    alpha: float = 0.7
    linewidth: float = 2.0
    point_size: float = 50.0
    show_entanglement_lines: bool = True
    show_density_matrix: bool = True
    show_correlation: bool = True
    save_animation: bool = False
    output_path: str = "entanglement_animation.mp4"

class QuantumEntanglementViewer:
    """
    Advanced visualization of quantum entanglement and correlations.
    """
    
    def __init__(self, config: Optional[EntanglementVizConfig] = None):
        """
        Initialize the quantum entanglement viewer.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = EntanglementVizConfig()
        
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup visualization style."""
        plt.style.use('seaborn')
        sns.set_palette(self.config.colormap)
    
    def plot_entanglement_network(self,
                                density_matrix: np.ndarray,
                                positions: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot the entanglement network between quantum states.
        
        Args:
            density_matrix: Density matrix
            positions: Optional node positions
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        n_states = density_matrix.shape[0]
        
        # Compute positions if not provided
        if positions is None:
            positions = np.random.rand(n_states, 3)
        
        # Plot nodes
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=np.diag(density_matrix),
                           s=self.config.point_size,
                           cmap=self.config.colormap,
                           alpha=self.config.alpha)
        
        # Plot entanglement lines
        if self.config.show_entanglement_lines:
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    if abs(density_matrix[i, j]) > 0.1:  # Threshold for entanglement
                        ax.plot([positions[i, 0], positions[j, 0]],
                              [positions[i, 1], positions[j, 1]],
                              [positions[i, 2], positions[j, 2]],
                              'k-',
                              alpha=self.config.alpha * abs(density_matrix[i, j]),
                              linewidth=self.config.linewidth)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Population")
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Entanglement Network")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("entanglement_network_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def create_entanglement_animation(self,
                                   density_sequence: List[np.ndarray],
                                   positions: Optional[np.ndarray] = None) -> FuncAnimation:
        """
        Create an animation of entanglement evolution.
        
        Args:
            density_sequence: Sequence of density matrices
            positions: Optional node positions
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        n_states = density_sequence[0].shape[0]
        
        # Compute positions if not provided
        if positions is None:
            positions = np.random.rand(n_states, 3)
        
        def update(frame):
            ax.clear()
            
            # Get current density matrix
            density = density_sequence[frame]
            
            # Plot nodes
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=np.diag(density),
                               s=self.config.point_size,
                               cmap=self.config.colormap,
                               alpha=self.config.alpha)
            
            # Plot entanglement lines
            if self.config.show_entanglement_lines:
                for i in range(n_states):
                    for j in range(i + 1, n_states):
                        if abs(density[i, j]) > 0.1:  # Threshold for entanglement
                            ax.plot([positions[i, 0], positions[j, 0]],
                                  [positions[i, 1], positions[j, 1]],
                                  [positions[i, 2], positions[j, 2]],
                                  'k-',
                                  alpha=self.config.alpha * abs(density[i, j]),
                                  linewidth=self.config.linewidth)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label="Population")
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Frame {frame}")
            
            return [scatter]
        
        anim = FuncAnimation(fig, update,
                           frames=len(density_sequence),
                           interval=1000/self.config.fps,
                           blit=True)
        
        if self.config.save_animation:
            anim.save(self.config.output_path,
                     writer='ffmpeg',
                     fps=self.config.fps,
                     dpi=self.config.dpi)
            logger.info("entanglement_animation_saved",
                       path=self.config.output_path)
        
        return anim
    
    def plot_density_matrix(self,
                          density_matrix: np.ndarray,
                          save_path: Optional[str] = None) -> None:
        """
        Plot the density matrix with entanglement information.
        
        Args:
            density_matrix: Density matrix
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot density matrix
        im = ax.imshow(np.abs(density_matrix),
                      cmap=self.config.colormap,
                      alpha=self.config.alpha)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Magnitude")
        
        ax.set_xlabel("State")
        ax.set_ylabel("State")
        ax.set_title("Density Matrix")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("density_matrix_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def plot_correlation_matrix(self,
                             density_matrix: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot the correlation matrix between quantum states.
        
        Args:
            density_matrix: Density matrix
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Compute correlation matrix
        correlation = np.zeros_like(density_matrix)
        n_states = density_matrix.shape[0]
        
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    correlation[i, j] = abs(density_matrix[i, j]) / np.sqrt(
                        density_matrix[i, i] * density_matrix[j, j])
        
        # Plot correlation matrix
        im = ax.imshow(correlation,
                      cmap=self.config.colormap,
                      alpha=self.config.alpha)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Correlation")
        
        ax.set_xlabel("State")
        ax.set_ylabel("State")
        ax.set_title("Correlation Matrix")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("correlation_matrix_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def plot_entanglement_spectrum(self,
                                 density_matrix: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot the entanglement spectrum (eigenvalues of the density matrix).
        
        Args:
            density_matrix: Density matrix
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(density_matrix)
        eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort in descending order
        
        # Plot spectrum
        ax.plot(range(1, len(eigenvalues) + 1), eigenvalues,
               'o-', alpha=self.config.alpha,
               linewidth=self.config.linewidth)
        
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Entanglement Spectrum")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("entanglement_spectrum_plot_saved",
                       path=save_path)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create viewer
    config = EntanglementVizConfig(
        figsize=(10, 8),
        colormap="plasma",
        show_entanglement_lines=True,
        show_density_matrix=True,
        show_correlation=True
    )
    viewer = QuantumEntanglementViewer(config)
    
    # Generate sample data
    n_states = 5
    density_matrix = np.random.rand(n_states, n_states) + 1j * np.random.rand(n_states, n_states)
    density_matrix = (density_matrix + density_matrix.conj().T) / 2  # Make Hermitian
    density_matrix = density_matrix / np.trace(density_matrix)  # Normalize
    
    # Create sequence for animation
    n_frames = 50
    density_sequence = []
    
    for i in range(n_frames):
        # Add some dynamics
        phase = 2 * np.pi * i / n_frames
        density_dyn = density_matrix * np.exp(1j * phase)
        density_dyn = (density_dyn + density_dyn.conj().T) / 2
        density_dyn = density_dyn / np.trace(density_dyn)
        density_sequence.append(density_dyn)
    
    # Create visualizations
    viewer.plot_entanglement_network(density_matrix)
    anim = viewer.create_entanglement_animation(density_sequence)
    viewer.plot_density_matrix(density_matrix)
    viewer.plot_correlation_matrix(density_matrix)
    viewer.plot_entanglement_spectrum(density_matrix) 
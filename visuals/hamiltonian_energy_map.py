# ==========================
# visuals/hamiltonian_energy_map.py
# ==========================
# Advanced visualization of Hamiltonian energy landscapes.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("visuals.hamiltonian")

@dataclass
class EnergyMapConfig:
    """Configuration for Hamiltonian energy map visualization."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    fps: int = 30
    colormap: str = "viridis"
    alpha: float = 0.7
    contour_levels: int = 20
    show_contours: bool = True
    show_surface: bool = True
    show_minima: bool = True
    show_trajectory: bool = True
    save_animation: bool = False
    output_path: str = "hamiltonian_energy_animation.mp4"

class HamiltonianEnergyMap:
    """
    Advanced visualization of Hamiltonian energy landscapes and trajectories.
    """
    
    def __init__(self, config: Optional[EnergyMapConfig] = None):
        """
        Initialize the Hamiltonian energy map visualizer.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = EnergyMapConfig()
        
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup visualization style."""
        plt.style.use('seaborn')
        sns.set_palette(self.config.colormap)
    
    def plot_energy_surface(self,
                          x: np.ndarray,
                          y: np.ndarray,
                          energy: np.ndarray,
                          trajectory: Optional[np.ndarray] = None,
                          minima: Optional[List[Tuple[float, float]]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot the energy surface with optional trajectory and minima.
        
        Args:
            x: X coordinates
            y: Y coordinates
            energy: Energy values
            trajectory: Optional trajectory points
            minima: Optional list of minima coordinates
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        if self.config.show_surface:
            surf = ax.plot_surface(x, y, energy,
                                 cmap=self.config.colormap,
                                 alpha=self.config.alpha)
            fig.colorbar(surf, ax=ax, label="Energy")
        
        # Plot contours
        if self.config.show_contours:
            ax.contour(x, y, energy,
                      levels=self.config.contour_levels,
                      cmap=self.config.colormap,
                      alpha=self.config.alpha)
        
        # Plot trajectory
        if self.config.show_trajectory and trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   np.interp(trajectory[:, 0], x[0, :], energy[0, :]),
                   'r-', linewidth=2, alpha=self.config.alpha)
        
        # Plot minima
        if self.config.show_minima and minima is not None:
            for min_x, min_y in minima:
                min_z = np.interp(min_x, x[0, :], energy[0, :])
                ax.scatter(min_x, min_y, min_z,
                         color='red', s=100, marker='*')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Energy")
        ax.set_title("Hamiltonian Energy Surface")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("energy_surface_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def create_energy_animation(self,
                              x: np.ndarray,
                              y: np.ndarray,
                              energy_sequence: List[np.ndarray],
                              trajectory: Optional[np.ndarray] = None,
                              minima: Optional[List[Tuple[float, float]]] = None) -> FuncAnimation:
        """
        Create an animation of the energy surface evolution.
        
        Args:
            x: X coordinates
            y: Y coordinates
            energy_sequence: Sequence of energy surfaces
            trajectory: Optional trajectory points
            minima: Optional list of minima coordinates
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Plot current energy surface
            if self.config.show_surface:
                surf = ax.plot_surface(x, y, energy_sequence[frame],
                                     cmap=self.config.colormap,
                                     alpha=self.config.alpha)
                fig.colorbar(surf, ax=ax, label="Energy")
            
            # Plot contours
            if self.config.show_contours:
                ax.contour(x, y, energy_sequence[frame],
                          levels=self.config.contour_levels,
                          cmap=self.config.colormap,
                          alpha=self.config.alpha)
            
            # Plot trajectory up to current frame
            if self.config.show_trajectory and trajectory is not None:
                current_traj = trajectory[:frame+1]
                ax.plot(current_traj[:, 0], current_traj[:, 1],
                       np.interp(current_traj[:, 0], x[0, :], energy_sequence[frame][0, :]),
                       'r-', linewidth=2, alpha=self.config.alpha)
            
            # Plot minima
            if self.config.show_minima and minima is not None:
                for min_x, min_y in minima:
                    min_z = np.interp(min_x, x[0, :], energy_sequence[frame][0, :])
                    ax.scatter(min_x, min_y, min_z,
                             color='red', s=100, marker='*')
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Energy")
            ax.set_title(f"Frame {frame}")
            
            return [surf] if self.config.show_surface else []
        
        anim = FuncAnimation(fig, update,
                           frames=len(energy_sequence),
                           interval=1000/self.config.fps,
                           blit=True)
        
        if self.config.save_animation:
            anim.save(self.config.output_path,
                     writer='ffmpeg',
                     fps=self.config.fps,
                     dpi=self.config.dpi)
            logger.info("energy_animation_saved",
                       path=self.config.output_path)
        
        return anim
    
    def plot_energy_contours(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           energy: np.ndarray,
                           trajectory: Optional[np.ndarray] = None,
                           minima: Optional[List[Tuple[float, float]]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Plot energy contours with optional trajectory and minima.
        
        Args:
            x: X coordinates
            y: Y coordinates
            energy: Energy values
            trajectory: Optional trajectory points
            minima: Optional list of minima coordinates
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot contours
        contour = ax.contourf(x, y, energy,
                            levels=self.config.contour_levels,
                            cmap=self.config.colormap,
                            alpha=self.config.alpha)
        fig.colorbar(contour, ax=ax, label="Energy")
        
        # Plot trajectory
        if self.config.show_trajectory and trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   'r-', linewidth=2, alpha=self.config.alpha)
        
        # Plot minima
        if self.config.show_minima and minima is not None:
            for min_x, min_y in minima:
                ax.scatter(min_x, min_y,
                         color='red', s=100, marker='*')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Hamiltonian Energy Contours")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("energy_contours_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def plot_energy_histogram(self,
                            energies: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of energy values.
        
        Args:
            energies: Energy values
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Plot histogram
        sns.histplot(energies, kde=True,
                    color=sns.color_palette(self.config.colormap)[0],
                    alpha=self.config.alpha,
                    ax=ax)
        
        ax.set_xlabel("Energy")
        ax.set_ylabel("Count")
        ax.set_title("Energy Distribution")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("energy_histogram_plot_saved",
                       path=save_path)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create visualizer
    config = EnergyMapConfig(
        figsize=(10, 8),
        colormap="plasma",
        show_contours=True,
        show_surface=True,
        show_minima=True,
        show_trajectory=True
    )
    visualizer = HamiltonianEnergyMap(config)
    
    # Generate sample data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create energy surface
    energy = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)
    
    # Create trajectory
    t = np.linspace(0, 10, 100)
    trajectory = np.column_stack((
        np.sin(t),
        np.cos(t)
    ))
    
    # Find minima
    minima = [(0, 0), (np.pi, np.pi), (-np.pi, -np.pi)]
    
    # Create sequence for animation
    n_frames = 50
    energy_sequence = []
    
    for i in range(n_frames):
        # Add some dynamics
        phase = 2 * np.pi * i / n_frames
        energy_dyn = np.sin(X + phase) * np.cos(Y + phase) + 0.1 * (X**2 + Y**2)
        energy_sequence.append(energy_dyn)
    
    # Create visualizations
    visualizer.plot_energy_surface(X, Y, energy, trajectory, minima)
    anim = visualizer.create_energy_animation(X, Y, energy_sequence, trajectory, minima)
    visualizer.plot_energy_contours(X, Y, energy, trajectory, minima)
    visualizer.plot_energy_histogram(energy.flatten()) 
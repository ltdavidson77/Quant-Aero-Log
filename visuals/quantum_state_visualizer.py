# ==========================
# visuals/quantum_state_visualizer.py
# ==========================
# Advanced visualization of quantum states and their evolution.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("visuals.quantum")

@dataclass
class QuantumVizConfig:
    """Configuration for quantum state visualization."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    fps: int = 30
    trail_length: int = 50
    colormap: str = "viridis"
    alpha: float = 0.7
    linewidth: float = 2.0
    point_size: float = 50.0
    show_entanglement: bool = True
    show_probability: bool = True
    show_phase: bool = True
    save_animation: bool = False
    output_path: str = "quantum_state_animation.mp4"

class QuantumStateVisualizer:
    """
    Advanced visualization of quantum states with entanglement and phase information.
    """
    
    def __init__(self, config: Optional[QuantumVizConfig] = None):
        """
        Initialize the quantum state visualizer.
        
        Args:
            config: Optional configuration object
        """
        if config is None:
            config = QuantumVizConfig()
        
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup visualization style."""
        plt.style.use('seaborn')
        sns.set_palette(self.config.colormap)
    
    def plot_state_evolution(self,
                           states: List[np.ndarray],
                           times: np.ndarray,
                           entanglement: Optional[List[float]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Plot the evolution of quantum states over time.
        
        Args:
            states: List of quantum state vectors
            times: Array of time points
            entanglement: Optional list of entanglement measures
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize)
        
        # Plot state components
        for i in range(states[0].shape[0]):
            component = [state[i] for state in states]
            ax1.plot(times, np.real(component), 
                    label=f"Re($\psi_{i}$)",
                    alpha=self.config.alpha,
                    linewidth=self.config.linewidth)
            ax1.plot(times, np.imag(component), 
                    label=f"Im($\psi_{i}$)",
                    alpha=self.config.alpha,
                    linewidth=self.config.linewidth,
                    linestyle='--')
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)
        
        # Plot probabilities
        probabilities = [np.abs(state)**2 for state in states]
        for i in range(states[0].shape[0]):
            prob = [p[i] for p in probabilities]
            ax2.plot(times, prob,
                    label=f"$|\psi_{i}|^2$",
                    alpha=self.config.alpha,
                    linewidth=self.config.linewidth)
        
        if entanglement is not None:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(times, entanglement,
                         color='red',
                         label="Entanglement",
                         alpha=self.config.alpha,
                         linewidth=self.config.linewidth)
            ax2_twin.set_ylabel("Entanglement")
            ax2_twin.legend(loc='upper right')
        
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Probability")
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("state_evolution_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def create_3d_animation(self,
                          states: List[np.ndarray],
                          times: np.ndarray,
                          entanglement: Optional[List[float]] = None) -> FuncAnimation:
        """
        Create a 3D animation of quantum state evolution.
        
        Args:
            states: List of quantum state vectors
            times: Array of time points
            entanglement: Optional list of entanglement measures
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        points = ax.scatter([], [], [], 
                          c=[],
                          cmap=self.config.colormap,
                          s=self.config.point_size,
                          alpha=self.config.alpha)
        
        if entanglement is not None:
            lines = [ax.plot([], [], [], 
                           color='red',
                           alpha=self.config.alpha,
                           linewidth=self.config.linewidth)[0]
                    for _ in range(len(states[0]) - 1)]
        
        def update(frame):
            state = states[frame]
            x = np.real(state)
            y = np.imag(state)
            z = np.abs(state)**2
            
            # Update points
            points._offsets3d = (x, y, z)
            points.set_array(np.arange(len(state)))
            
            # Update entanglement lines
            if entanglement is not None and entanglement[frame] > 0.5:
                for i in range(len(state) - 1):
                    lines[i].set_data([x[i], x[i+1]], [y[i], y[i+1]])
                    lines[i].set_3d_properties([z[i], z[i+1]])
            
            ax.set_title(f"Time: {times[frame]:.2f}")
            return [points] + (lines if entanglement is not None else [])
        
        anim = FuncAnimation(fig, update,
                           frames=len(states),
                           interval=1000/self.config.fps,
                           blit=True)
        
        if self.config.save_animation:
            anim.save(self.config.output_path,
                     writer='ffmpeg',
                     fps=self.config.fps,
                     dpi=self.config.dpi)
            logger.info("quantum_state_animation_saved",
                       path=self.config.output_path)
        
        return anim
    
    def plot_phase_space(self,
                        states: List[np.ndarray],
                        times: np.ndarray,
                        save_path: Optional[str] = None) -> None:
        """
        Plot the phase space representation of quantum states.
        
        Args:
            states: List of quantum state vectors
            times: Array of time points
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Compute phase space coordinates
        q = [np.real(state) for state in states]
        p = [np.imag(state) for state in states]
        
        # Create phase space plot
        for i in range(states[0].shape[0]):
            qi = [q[j][i] for j in range(len(states))]
            pi = [p[j][i] for j in range(len(states))]
            
            # Plot trajectory
            ax.plot(qi, pi,
                   label=f"State {i}",
                   alpha=self.config.alpha,
                   linewidth=self.config.linewidth)
            
            # Add current position
            ax.scatter(qi[-1], pi[-1],
                      s=self.config.point_size,
                      alpha=self.config.alpha)
        
        ax.set_xlabel("Position (q)")
        ax.set_ylabel("Momentum (p)")
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("phase_space_plot_saved",
                       path=save_path)
        
        plt.show()
    
    def plot_entanglement_matrix(self,
                               states: List[np.ndarray],
                               save_path: Optional[str] = None) -> None:
        """
        Plot the entanglement matrix between quantum states.
        
        Args:
            states: List of quantum state vectors
            save_path: Optional path to save the figure
        """
        # Compute entanglement matrix
        n_states = len(states[0])
        entanglement = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(i + 1, n_states):
                # Compute correlation between states
                corr = np.corrcoef([s[i] for s in states],
                                 [s[j] for s in states])[0, 1]
                entanglement[i, j] = entanglement[j, i] = abs(corr)
        
        # Plot entanglement matrix
        fig, ax = plt.subplots(figsize=self.config.figsize)
        sns.heatmap(entanglement,
                   cmap=self.config.colormap,
                   annot=True,
                   fmt=".2f",
                   ax=ax)
        
        ax.set_title("Entanglement Matrix")
        ax.set_xlabel("State")
        ax.set_ylabel("State")
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi)
            logger.info("entanglement_matrix_plot_saved",
                       path=save_path)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create visualizer
    config = QuantumVizConfig(
        figsize=(10, 8),
        colormap="plasma",
        show_entanglement=True
    )
    visualizer = QuantumStateVisualizer(config)
    
    # Generate sample data
    times = np.linspace(0, 10, 100)
    states = []
    entanglement = []
    
    for t in times:
        # Create a simple quantum state
        state = np.array([
            np.exp(1j * t),
            np.exp(1j * 2 * t),
            np.exp(1j * 3 * t)
        ])
        states.append(state)
        
        # Compute entanglement
        ent = np.abs(np.corrcoef(np.real(state), np.imag(state))[0, 1])
        entanglement.append(ent)
    
    # Create visualizations
    visualizer.plot_state_evolution(states, times, entanglement)
    anim = visualizer.create_3d_animation(states, times, entanglement)
    visualizer.plot_phase_space(states, times)
    visualizer.plot_entanglement_matrix(states) 
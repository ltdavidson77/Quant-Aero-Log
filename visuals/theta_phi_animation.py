# ==========================
# visuals/theta_phi_animation.py
# ==========================
# Animated 2D path of theta vs phi evolution over time.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Union, Optional, Tuple
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class ThetaPhiAnimator:
    def __init__(self,
                 figsize: Tuple[int, int] = (10, 8),
                 style: str = 'seaborn',
                 color_palette: str = 'viridis'):
        """
        Initialize the theta-phi animator.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style to use
            color_palette: Seaborn color palette name
        """
        self.figsize = figsize
        self.style = style
        self.color_palette = color_palette
        plt.style.use(style)
        sns.set_palette(color_palette)
    
    def create_animation(self,
                        theta_series: np.ndarray,
                        phi_series: np.ndarray,
                        interval: int = 100,
                        title: str = "Theta vs Phi Evolution",
                        save_path: Optional[str] = None,
                        show_trail: bool = True,
                        trail_length: int = 50,
                        add_heatmap: bool = True) -> None:
        """
        Create an animated visualization of theta vs phi evolution.
        
        Args:
            theta_series: Array of theta values
            phi_series: Array of phi values
            interval: Animation interval in milliseconds
            title: Plot title
            save_path: Optional path to save the animation
            show_trail: Whether to show the path trail
            trail_length: Length of the trail to show
            add_heatmap: Whether to add a background heatmap
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Set axis limits with some padding
        x_pad = (max(theta_series) - min(theta_series)) * 0.1
        y_pad = (max(phi_series) - min(phi_series)) * 0.1
        ax.set_xlim(min(theta_series) - x_pad, max(theta_series) + x_pad)
        ax.set_ylim(min(phi_series) - y_pad, max(phi_series) + y_pad)
        
        # Add heatmap if requested
        if add_heatmap:
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(theta_series, phi_series, bins=50)
            H = H.T  # Transpose for correct orientation
            
            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list(
                'custom', ['#ffffff', '#0000ff'], N=256)
            
            # Plot heatmap
            ax.imshow(H, interpolation='nearest', origin='lower',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                     aspect='auto', cmap=cmap, alpha=0.3)
        
        # Initialize line and scatter plot
        line, = ax.plot([], [], 'b-', lw=2, alpha=0.5)
        scatter = ax.scatter([], [], c='red', s=50, alpha=0.8)
        
        # Initialize trail
        trail_x, trail_y = [], []
        
        def update(frame):
            nonlocal trail_x, trail_y
            
            # Update trail
            if show_trail:
                trail_x.append(theta_series[frame])
                trail_y.append(phi_series[frame])
                
                # Keep only the last trail_length points
                if len(trail_x) > trail_length:
                    trail_x = trail_x[-trail_length:]
                    trail_y = trail_y[-trail_length:]
                
                line.set_data(trail_x, trail_y)
            
            # Update current position
            scatter.set_offsets(np.c_[theta_series[frame], phi_series[frame]])
            
            return line, scatter
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update,
            frames=len(theta_series),
            interval=interval,
            blit=True
        )
        
        # Add labels and title
        ax.set_xlabel("Theta (Gradient Angle)")
        ax.set_ylabel("Phi (MA Angle)")
        ax.set_title(title, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save animation if path provided
        if save_path:
            ani.save(save_path, writer='pillow', fps=30)
        
        plt.tight_layout()
        plt.show()
    
    def plot_static_path(self,
                        theta_series: np.ndarray,
                        phi_series: np.ndarray,
                        title: str = "Theta vs Phi Path",
                        save_path: Optional[str] = None,
                        add_heatmap: bool = True) -> None:
        """
        Create a static visualization of the theta-phi path.
        
        Args:
            theta_series: Array of theta values
            phi_series: Array of phi values
            title: Plot title
            save_path: Optional path to save the figure
            add_heatmap: Whether to add a background heatmap
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Add heatmap if requested
        if add_heatmap:
            H, xedges, yedges = np.histogram2d(theta_series, phi_series, bins=50)
            H = H.T
            
            cmap = LinearSegmentedColormap.from_list(
                'custom', ['#ffffff', '#0000ff'], N=256)
            
            ax.imshow(H, interpolation='nearest', origin='lower',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                     aspect='auto', cmap=cmap, alpha=0.3)
        
        # Plot path
        ax.plot(theta_series, phi_series, 'b-', lw=2, alpha=0.5)
        ax.scatter(theta_series[-1], phi_series[-1], c='red', s=50)
        
        # Add labels and title
        ax.set_xlabel("Theta (Gradient Angle)")
        ax.set_ylabel("Phi (MA Angle)")
        ax.set_title(title, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage
if __name__ == "__main__":
    from generate_data import get_price_series
    from compute_analytics import compute_angles
    
    # Generate sample data
    df = get_price_series()
    theta, phi, _, _ = compute_angles(df)
    
    # Create animator instance
    animator = ThetaPhiAnimator()
    
    # Create animation
    animator.create_animation(
        theta[:300],
        phi[:300],
        interval=50,
        show_trail=True,
        trail_length=30,
        add_heatmap=True
    )
    
    # Create static plot
    animator.plot_static_path(
        theta[:300],
        phi[:300],
        add_heatmap=True
    )

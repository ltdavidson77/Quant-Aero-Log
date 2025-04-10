# ==========================
# visuals/volatility_cloud.py
# ==========================
# 3D Surface Plot of volatility vs angle metrics.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional
import matplotlib.colors as mcolors
from matplotlib import cm

class VolatilityCloud:
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 cmap: str = 'viridis',
                 alpha: float = 0.8,
                 edge_color: str = 'none'):
        self.figsize = figsize
        self.cmap = cmap
        self.alpha = alpha
        self.edge_color = edge_color
        
    def plot(self, 
             theta: np.ndarray, 
             phi: np.ndarray, 
             vol: np.ndarray,
             title: str = "Volatility Cloud",
             elev: int = 20,
             azim: int = 45,
             save_path: Optional[str] = None) -> None:
        """
        Create a 3D surface plot of volatility vs angle metrics.
        
        Args:
            theta: Array of theta (gradient) angles
            phi: Array of phi (MA) angles
            vol: Array of volatility values
            title: Plot title
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            save_path: Optional path to save the figure
        """
        # Create figure and 3D axis
        fig = plt.figure(figsize=self.figsize)
    ax = fig.add_subplot(111, projection='3d')
        
        # Normalize volatility for coloring
        norm = mcolors.Normalize(vmin=vol.min(), vmax=vol.max())
        colors = cm.get_cmap(self.cmap)(norm(vol))
        
        # Create surface plot
        surf = ax.plot_trisurf(
            theta, phi, vol,
            cmap=self.cmap,
            edgecolor=self.edge_color,
            alpha=self.alpha,
            linewidth=0.2
        )
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Volatility')
        
        # Set labels and title
        ax.set_xlabel("Theta (Gradient Angle)", labelpad=10)
        ax.set_ylabel("Phi (MA Angle)", labelpad=10)
        ax.set_zlabel("Volatility", labelpad=10)
        ax.set_title(title, pad=20)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_contour(self,
                    theta: np.ndarray,
                    phi: np.ndarray,
                    vol: np.ndarray,
                    title: str = "Volatility Contour",
                    levels: int = 20,
                    save_path: Optional[str] = None) -> None:
        """
        Create a 2D contour plot of volatility vs angle metrics.
        
        Args:
            theta: Array of theta (gradient) angles
            phi: Array of phi (MA) angles
            vol: Array of volatility values
            title: Plot title
            levels: Number of contour levels
            save_path: Optional path to save the figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create contour plot
        contour = ax.tricontourf(
            theta, phi, vol,
            levels=levels,
            cmap=self.cmap
        )
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Volatility')
        
        # Set labels and title
    ax.set_xlabel("Theta (Gradient Angle)")
    ax.set_ylabel("Phi (MA Angle)")
    ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
    plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

# Example usage
if __name__ == "__main__":
    from generate_data import get_price_series
    from compute_analytics import compute_angles

    # Generate sample data
    df = get_price_series()
    theta, phi, _, vol = compute_angles(df)
    
    # Limit data points for better visualization
    n_points = 500
    theta = theta[:n_points]
    phi = phi[:n_points]
    vol = vol[:n_points]
    
    # Create and plot volatility cloud
    vc = VolatilityCloud()
    vc.plot(theta, phi, vol, title="Volatility Cloud Analysis")
    
    # Create and plot contour
    vc.plot_contour(theta, phi, vol, title="Volatility Contour Analysis")

# ==========================
# visuals/prediction_overlay.py
# ==========================
# Plot predicted vs actual values as line overlays with advanced features.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
import seaborn as sns
from matplotlib.widgets import Slider, Button
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from datetime import datetime

class PredictionOverlay:
    def __init__(self,
                 figsize: Tuple[int, int] = (14, 8),
                 style: str = 'seaborn',
                 color_palette: str = 'viridis'):
        """
        Initialize the prediction overlay visualizer.
        
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
    
    def plot(self,
             df: pd.DataFrame,
             actual_col: str,
             pred_col: str,
             title: str = "Prediction Overlay",
             save_path: Optional[str] = None,
             show_confidence: bool = True,
             confidence_col: Optional[str] = None,
             show_error: bool = True,
             error_window: int = 20,
             interactive: bool = False) -> None:
        """
        Create an enhanced prediction overlay plot.
        
        Args:
            df: DataFrame containing the data
            actual_col: Column name for actual values
            pred_col: Column name for predicted values
            title: Plot title
            save_path: Optional path to save the figure
            show_confidence: Whether to show confidence intervals
            confidence_col: Column name for confidence values
            show_error: Whether to show error metrics
            error_window: Window size for error calculation
            interactive: Whether to add interactive features
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual values
        actual_line = ax.plot(df.index, df[actual_col],
                            label="Actual",
                            color='black',
                            alpha=0.7,
                            linewidth=2)[0]
        
        # Plot predicted values
        pred_line = ax.plot(df.index, df[pred_col],
                          label="Predicted",
                          linestyle='--',
                          color='red',
                          alpha=0.8,
                          linewidth=2)[0]
        
        # Add confidence intervals if available
        if show_confidence and confidence_col is not None:
            confidence = df[confidence_col]
            ax.fill_between(df.index,
                          df[pred_col] - confidence,
                          df[pred_col] + confidence,
                          color='red',
                          alpha=0.2,
                          label='95% Confidence Interval')
        
        # Calculate and plot error metrics if requested
        if show_error:
            error = df[actual_col] - df[pred_col]
            rolling_error = error.rolling(window=error_window).mean()
            rolling_std = error.rolling(window=error_window).std()
            
            # Add error subplot
            ax_error = ax.twinx()
            error_line = ax_error.plot(df.index, rolling_error,
                                     label="Prediction Error",
                                     color='blue',
                                     alpha=0.6,
                                     linewidth=1)[0]
            
            # Add error confidence interval
            ax_error.fill_between(df.index,
                                rolling_error - 2*rolling_std,
                                rolling_error + 2*rolling_std,
                                color='blue',
                                alpha=0.1)
            
            ax_error.set_ylabel("Prediction Error", color='blue')
            ax_error.tick_params(axis='y', labelcolor='blue')
            ax_error.grid(False)
        
        # Customize main plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal Value")
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
        
        # Add legend
        lines = [actual_line, pred_line]
        if show_error:
            lines.append(error_line)
        ax.legend(lines, [l.get_label() for l in lines], loc='best')
        
        # Add interactive features if requested
        if interactive:
            self._add_interactive_features(fig, ax, df, actual_col, pred_col)
        
    plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    def _add_interactive_features(self,
                                fig: plt.Figure,
                                ax: plt.Axes,
                                df: pd.DataFrame,
                                actual_col: str,
                                pred_col: str) -> None:
        """Add interactive features to the plot."""
        # Add zoom slider
        ax_zoom = plt.axes([0.2, 0.01, 0.65, 0.03])
        zoom_slider = Slider(ax_zoom, 'Zoom', 0.1, 1.0, valinit=1.0)
        
        # Add reset button
        ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
        reset_button = Button(ax_reset, 'Reset')
        
        def update_zoom(val):
            zoom = zoom_slider.val
            x_range = df.index[-1] - df.index[0]
            center = df.index[0] + x_range/2
            new_range = x_range * zoom
            ax.set_xlim(center - new_range/2, center + new_range/2)
            fig.canvas.draw_idle()
        
        def reset_view(event):
            zoom_slider.reset()
            ax.set_xlim(df.index[0], df.index[-1])
            fig.canvas.draw_idle()
        
        zoom_slider.on_changed(update_zoom)
        reset_button.on_clicked(reset_view)
    
    def plot_prediction_metrics(self,
                              df: pd.DataFrame,
                              actual_col: str,
                              pred_col: str,
                              title: str = "Prediction Metrics",
                              save_path: Optional[str] = None) -> None:
        """
        Plot various prediction metrics and statistics.
        
        Args:
            df: DataFrame containing the data
            actual_col: Column name for actual values
            pred_col: Column name for predicted values
            title: Plot title
            save_path: Optional path to save the figure
        """
        # Calculate metrics
        error = df[actual_col] - df[pred_col]
        mae = np.abs(error).mean()
        mse = (error ** 2).mean()
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(error / df[actual_col])) * 100
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot error distribution
        sns.histplot(error, ax=ax1, kde=True)
        ax1.set_title("Error Distribution")
        ax1.set_xlabel("Prediction Error")
        ax1.set_ylabel("Frequency")
        
        # Plot cumulative error
        cumulative_error = error.cumsum()
        ax2.plot(df.index, cumulative_error)
        ax2.set_title("Cumulative Error")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Cumulative Error")
        
        # Plot error over time
        rolling_error = error.rolling(window=20).mean()
        ax3.plot(df.index, rolling_error)
        ax3.set_title("Rolling Mean Error")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Error")
        
        # Add metrics text
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%"
        ax4.text(0.5, 0.5, metrics_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax4.transAxes,
                fontsize=12)
        ax4.axis('off')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create overlay instance
    overlay = PredictionOverlay()
    
    # Generate sample data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    actual = np.random.normal(0, 1, 100).cumsum() + 100
    predicted = actual + np.random.normal(0, 0.5, 100)
    confidence = np.random.uniform(0.1, 0.3, 100)
    
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted,
        'confidence': confidence
    }, index=dates)
    
    # Plot prediction overlay
    overlay.plot(df,
                actual_col='actual',
                pred_col='predicted',
                confidence_col='confidence',
                show_error=True,
                interactive=True)
    
    # Plot prediction metrics
    overlay.plot_prediction_metrics(df,
                                  actual_col='actual',
                                  pred_col='predicted')

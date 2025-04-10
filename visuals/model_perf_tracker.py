# ==========================
# visuals/model_perf_tracker.py
# ==========================
# Track model accuracy and class distribution over time.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import PercentFormatter

class ModelPerformanceTracker:
    def __init__(self, 
                 figsize: tuple = (12, 6),
                 style: str = 'seaborn',
                 color_palette: str = 'viridis'):
        """
        Initialize the model performance tracker.
        
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
    
    def plot_accuracy_trend(self,
                          score_log: pd.DataFrame,
                          title: str = "Model Accuracy Over Time",
                          save_path: Optional[str] = None,
                          show_confidence: bool = True,
                          window_size: int = 5) -> None:
        """
        Plot model accuracy trend with confidence intervals.
        
        Args:
            score_log: DataFrame with timestamp and accuracy columns
            title: Plot title
            save_path: Optional path to save the figure
            show_confidence: Whether to show confidence intervals
            window_size: Rolling window size for confidence calculation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate rolling statistics
        if show_confidence and len(score_log) > window_size:
            rolling_mean = score_log['accuracy'].rolling(window=window_size).mean()
            rolling_std = score_log['accuracy'].rolling(window=window_size).std()
            
            # Plot confidence intervals
            ax.fill_between(score_log['timestamp'],
                          rolling_mean - 2*rolling_std,
                          rolling_mean + 2*rolling_std,
                          alpha=0.2,
                          label='95% Confidence Interval')
        
        # Plot accuracy line
        ax.plot(score_log['timestamp'], 
                score_log['accuracy'],
                marker='o',
                markersize=4,
                linewidth=2,
                label='Accuracy')
        
        # Add horizontal line at 50% for reference
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random Guess')
        
        # Customize plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self,
                              predictions: Union[pd.Series, np.ndarray],
                              title: str = "Prediction Class Distribution",
                              save_path: Optional[str] = None,
                              normalize: bool = False) -> None:
        """
        Plot class distribution with additional statistics.
        
        Args:
            predictions: Series or array of predictions
            title: Plot title
            save_path: Optional path to save the figure
            normalize: Whether to show percentages instead of counts
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to Series if numpy array
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
        
        # Calculate class counts
        class_counts = predictions.value_counts().sort_index()
        
        if normalize:
            class_counts = class_counts / class_counts.sum()
            ylabel = "Percentage"
            fmt = '{:.1%}'
        else:
            ylabel = "Count"
            fmt = '{:d}'
        
        # Create bar plot
        bars = ax.bar(class_counts.index, class_counts.values, color=sns.color_palette())
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   fmt.format(height),
                   ha='center', va='bottom')
        
        # Customize plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Class")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if normalize:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self,
                            y_true: Union[np.ndarray, List],
                            y_pred: Union[np.ndarray, List],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_path: Optional path to save the figure
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Customize plot
        ax.set_title(title, pad=20)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create tracker instance
    tracker = ModelPerformanceTracker()
    
    # Generate sample data
    df_log = pd.DataFrame({
        "timestamp": pd.date_range(start='2025-01-01', periods=20, freq='H'),
        "accuracy": np.random.normal(0.7, 0.05, 20).clip(0, 1)
    })
    
    # Plot accuracy trend
    tracker.plot_accuracy_trend(df_log, show_confidence=True)
    
    # Generate sample predictions
    preds = np.random.randint(0, 3, 100)
    
    # Plot class distribution
    tracker.plot_class_distribution(preds, normalize=True)
    
    # Generate sample true and predicted labels
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    
    # Plot confusion matrix
    tracker.plot_confusion_matrix(y_true, y_pred)

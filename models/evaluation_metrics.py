# ==========================
# evaluation_metrics.py
# ==========================
# Contains financial metrics and evaluation functions.

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import time
import gc
from pathlib import Path
import sys

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

logger = logging.getLogger(__name__)

class FinancialMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 returns: np.ndarray = None, prices: np.ndarray = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.returns = returns
        self.prices = prices
        
    def compute_classification_metrics(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Compute standard classification metrics."""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        accuracy = accuracy_score(self.y_true, self.y_pred)
        mcc = matthews_corrcoef(self.y_true, self.y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        metrics_df.index.name = 'Class'
        
        summary = pd.Series({
            'Accuracy': accuracy,
            'MCC': mcc,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp
        })
        
        return metrics_df, summary
        
    def compute_financial_metrics(self) -> Dict[str, float]:
        """Compute financial performance metrics."""
        if self.returns is None or self.prices is None:
            logging.warning("Returns or prices not provided for financial metrics")
            return {}
            
        # Calculate returns for predictions
        pred_returns = self.returns[self.y_pred == 1]  # Assuming 1 is the positive class
        true_returns = self.returns[self.y_true == 1]
        
        # Basic financial metrics
        metrics = {
            'Predicted_Returns_Mean': np.mean(pred_returns) if len(pred_returns) > 0 else 0,
            'Predicted_Returns_Std': np.std(pred_returns) if len(pred_returns) > 0 else 0,
            'True_Returns_Mean': np.mean(true_returns) if len(true_returns) > 0 else 0,
            'True_Returns_Std': np.std(true_returns) if len(true_returns) > 0 else 0,
            'Sharpe_Ratio': (np.mean(pred_returns) / np.std(pred_returns)) if len(pred_returns) > 0 else 0,
            'Max_Drawdown': self._calculate_max_drawdown(self.prices),
            'Win_Rate': np.mean(pred_returns > 0) if len(pred_returns) > 0 else 0,
            'Profit_Factor': self._calculate_profit_factor(pred_returns)
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown from price series."""
        if len(prices) == 0:
            return 0.0
        peak = prices[0]
        max_drawdown = 0.0
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown
        
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if len(returns) == 0:
            return 0.0
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))
        return gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a single dictionary."""
        classification_metrics, summary = self.compute_classification_metrics()
        financial_metrics = self.compute_financial_metrics()
        
        return {
            'classification_metrics': classification_metrics,
            'summary': summary,
            'financial_metrics': financial_metrics
        }

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                          returns: np.ndarray = None,
                          prices: np.ndarray = None) -> Dict[str, Any]:
    """Main evaluation function that combines all metrics."""
    evaluator = FinancialMetrics(y_true, y_pred, returns, prices)
    return evaluator.get_all_metrics()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, size=100)
    y_pred = np.random.randint(0, 3, size=100)
    returns = np.random.normal(0.001, 0.02, size=100)
    prices = np.cumprod(1 + returns) * 100
    
    # Evaluate
    metrics = evaluate_classification(y_true, y_pred, returns, prices)
    
    print("Classification Metrics:")
    print(metrics['classification_metrics'])
    print("\nSummary:")
    print(metrics['summary'])
    print("\nFinancial Metrics:")
    print(metrics['financial_metrics'])

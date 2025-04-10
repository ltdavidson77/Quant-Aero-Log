# ==========================
# evaluation_metrics.py
# ==========================
# Contains comprehensive financial metrics and evaluation functions for stock analysis.

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support
)
import time
import gc
from pathlib import Path
import sys
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
from datetime import datetime, timedelta

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

# Import model management
from .model_manager import ModelManager, ModelError, ModelSaveError, ModelLoadError
from .model_loader import ModelLoader, ModelLoaderError

logger = logging.getLogger(__name__)

class MetricsError(Exception):
    """Base exception for metrics operations."""
    pass

class ClassificationMetricsError(MetricsError):
    """Exception raised when classification metrics calculation fails."""
    pass

class FinancialMetricsError(MetricsError):
    """Exception raised when financial metrics calculation fails."""
    pass

@dataclass
class StockMetrics:
    """Container for stock-specific metrics."""
    ticker: str
    start_date: datetime
    end_date: datetime
    benchmark: str = 'SPY'  # Default benchmark
    risk_free_rate: float = 0.02  # Annual risk-free rate

class FinancialMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 returns: np.ndarray = None, prices: np.ndarray = None,
                 stock_metrics: Optional[StockMetrics] = None,
                 benchmark_returns: np.ndarray = None,
                 model_manager: Optional[ModelManager] = None):
        try:
            self.y_true = y_true
            self.y_pred = y_pred
            self.returns = returns
            self.prices = prices
            self.stock_metrics = stock_metrics
            self.benchmark_returns = benchmark_returns
            self.model_manager = model_manager or ModelManager()
            self.model_loader = ModelLoader(self.model_manager)
        except Exception as e:
            raise MetricsError(f"Failed to initialize FinancialMetrics: {str(e)}")
        
    def compute_classification_metrics(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Compute standard classification metrics."""
        try:
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
        except Exception as e:
            raise ClassificationMetricsError(f"Failed to compute classification metrics: {str(e)}")
        
    def compute_financial_metrics(self) -> Dict[str, float]:
        """Compute comprehensive financial performance metrics."""
        try:
            if self.returns is None or self.prices is None:
                logger.warning("Returns or prices not provided for financial metrics")
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
                'Sharpe_Ratio': self._calculate_sharpe_ratio(pred_returns),
                'Sortino_Ratio': self._calculate_sortino_ratio(pred_returns),
                'Calmar_Ratio': self._calculate_calmar_ratio(pred_returns),
                'Max_Drawdown': self._calculate_max_drawdown(self.prices),
                'Win_Rate': np.mean(pred_returns > 0) if len(pred_returns) > 0 else 0,
                'Profit_Factor': self._calculate_profit_factor(pred_returns),
                'Average_Win': np.mean(pred_returns[pred_returns > 0]) if len(pred_returns[pred_returns > 0]) > 0 else 0,
                'Average_Loss': np.mean(pred_returns[pred_returns < 0]) if len(pred_returns[pred_returns < 0]) > 0 else 0,
                'Risk_Adjusted_Return': self._calculate_risk_adjusted_return(pred_returns),
                'Value_at_Risk': self._calculate_var(pred_returns),
                'Expected_Shortfall': self._calculate_expected_shortfall(pred_returns),
                'Skewness': skew(pred_returns) if len(pred_returns) > 0 else 0,
                'Kurtosis': kurtosis(pred_returns) if len(pred_returns) > 0 else 0,
                'Information_Ratio': self._calculate_information_ratio(pred_returns),
                'Tracking_Error': self._calculate_tracking_error(pred_returns),
                'Beta': self._calculate_beta(pred_returns),
                'Alpha': self._calculate_alpha(pred_returns),
                'Treynor_Ratio': self._calculate_treynor_ratio(pred_returns),
                'Jensen_Alpha': self._calculate_jensen_alpha(pred_returns),
                'Upside_Potential_Ratio': self._calculate_upside_potential_ratio(pred_returns),
                'Omega_Ratio': self._calculate_omega_ratio(pred_returns),
                'Tail_Ratio': self._calculate_tail_ratio(pred_returns),
                'Common_Sense_Ratio': self._calculate_common_sense_ratio(pred_returns),
                'Gain_Loss_Ratio': self._calculate_gain_loss_ratio(pred_returns),
                'Win_Loss_Ratio': self._calculate_win_loss_ratio(pred_returns),
                'Payoff_Ratio': self._calculate_payoff_ratio(pred_returns),
                'Profit_Risk_Ratio': self._calculate_profit_risk_ratio(pred_returns),
                'Average_Holding_Period': self._calculate_average_holding_period(),
                'Turnover_Ratio': self._calculate_turnover_ratio(),
                'Market_Timing_Score': self._calculate_market_timing_score(),
                'Volatility_Score': self._calculate_volatility_score(pred_returns),
                'Liquidity_Score': self._calculate_liquidity_score(),
                'Momentum_Score': self._calculate_momentum_score(pred_returns),
                'Quality_Score': self._calculate_quality_score(pred_returns),
                'Value_Score': self._calculate_value_score(pred_returns),
                'Growth_Score': self._calculate_growth_score(pred_returns),
                'Size_Score': self._calculate_size_score(),
                'Sentiment_Score': self._calculate_sentiment_score(),
                'Technical_Score': self._calculate_technical_score(pred_returns),
                'Fundamental_Score': self._calculate_fundamental_score(),
                'Risk_Score': self._calculate_risk_score(pred_returns),
                'Performance_Score': self._calculate_performance_score(pred_returns)
            }
            
            return metrics
        except Exception as e:
            raise FinancialMetricsError(f"Failed to compute financial metrics: {str(e)}")
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            excess_returns = returns - (self.stock_metrics.risk_free_rate / 252)  # Daily risk-free rate
            return np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Sharpe ratio: {str(e)}")
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            excess_returns = returns - (self.stock_metrics.risk_free_rate / 252)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            return np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Sortino ratio: {str(e)}")
        
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            annual_return = np.mean(returns) * 252
            max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + returns))
            return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Calmar ratio: {str(e)}")
        
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown from price series."""
        try:
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
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate max drawdown: {str(e)}")
        
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        try:
            if len(returns) == 0:
                return 0.0
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
            return gross_profits / gross_losses if gross_losses > 0 else float('inf')
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate profit factor: {str(e)}")
        
    def _calculate_risk_adjusted_return(self, returns: np.ndarray) -> float:
        """Calculate risk-adjusted return."""
        try:
            if len(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate risk-adjusted return: {str(e)}")
        
    def _calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        try:
            if len(returns) == 0:
                return 0.0
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Value at Risk: {str(e)}")
        
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            if len(returns) == 0:
                return 0.0
            var = self._calculate_var(returns, confidence_level)
            return np.mean(returns[returns <= var])
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Expected Shortfall: {str(e)}")
        
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information Ratio."""
        try:
            if len(returns) == 0 or self.benchmark_returns is None:
                return 0.0
            excess_returns = returns - self.benchmark_returns
            return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Information Ratio: {str(e)}")
        
    def _calculate_tracking_error(self, returns: np.ndarray) -> float:
        """Calculate Tracking Error."""
        try:
            if len(returns) == 0 or self.benchmark_returns is None:
                return 0.0
            excess_returns = returns - self.benchmark_returns
            return np.std(excess_returns)
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Tracking Error: {str(e)}")
        
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate Beta."""
        try:
            if len(returns) == 0 or self.benchmark_returns is None:
                return 0.0
            covariance = np.cov(returns, self.benchmark_returns)[0, 1]
            benchmark_variance = np.var(self.benchmark_returns)
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Beta: {str(e)}")
        
    def _calculate_alpha(self, returns: np.ndarray) -> float:
        """Calculate Alpha."""
        try:
            if len(returns) == 0 or self.benchmark_returns is None:
                return 0.0
            beta = self._calculate_beta(returns)
            excess_returns = returns - self.benchmark_returns
            return np.mean(excess_returns) - beta * np.mean(self.benchmark_returns)
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Alpha: {str(e)}")
        
    def _calculate_treynor_ratio(self, returns: np.ndarray) -> float:
        """Calculate Treynor Ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            beta = self._calculate_beta(returns)
            excess_returns = returns - (self.stock_metrics.risk_free_rate / 252)
            return np.mean(excess_returns) / beta if beta > 0 else 0
        except Exception as e:
            raise FinancialMetricsError(f"Failed to calculate Treynor Ratio: {str(e)}")
        
    def _calculate_jensen_alpha(self, returns: np.ndarray) -> float:
        """Calculate Jensen's Alpha."""
        if len(returns) == 0 or self.benchmark_returns is None:
            return 0.0
        beta = self._calculate_beta(returns)
        risk_free_rate = self.stock_metrics.risk_free_rate / 252
        expected_return = risk_free_rate + beta * (np.mean(self.benchmark_returns) - risk_free_rate)
        return np.mean(returns) - expected_return
        
    def _calculate_upside_potential_ratio(self, returns: np.ndarray) -> float:
        """Calculate Upside Potential Ratio."""
        if len(returns) == 0:
            return 0.0
        upside_returns = returns[returns > 0]
        downside_returns = returns[returns < 0]
        return np.mean(upside_returns) / np.std(downside_returns) if len(downside_returns) > 0 else 0
        
    def _calculate_omega_ratio(self, returns: np.ndarray) -> float:
        """Calculate Omega Ratio."""
        if len(returns) == 0:
            return 0.0
        threshold = 0
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        return np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 else float('inf')
        
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate Tail Ratio."""
        if len(returns) == 0:
            return 0.0
        return abs(np.percentile(returns, 5)) / abs(np.percentile(returns, 95))
        
    def _calculate_common_sense_ratio(self, returns: np.ndarray) -> float:
        """Calculate Common Sense Ratio."""
        if len(returns) == 0:
            return 0.0
        return self._calculate_omega_ratio(returns) * self._calculate_tail_ratio(returns)
        
    def _calculate_gain_loss_ratio(self, returns: np.ndarray) -> float:
        """Calculate Gain-Loss Ratio."""
        if len(returns) == 0:
            return 0.0
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        return np.mean(gains) / abs(np.mean(losses)) if len(losses) > 0 else float('inf')
        
    def _calculate_win_loss_ratio(self, returns: np.ndarray) -> float:
        """Calculate Win-Loss Ratio."""
        if len(returns) == 0:
            return 0.0
        wins = len(returns[returns > 0])
        losses = len(returns[returns < 0])
        return wins / losses if losses > 0 else float('inf')
        
    def _calculate_payoff_ratio(self, returns: np.ndarray) -> float:
        """Calculate Payoff Ratio."""
        if len(returns) == 0:
            return 0.0
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        return np.mean(gains) / abs(np.mean(losses)) if len(losses) > 0 else float('inf')
        
    def _calculate_profit_risk_ratio(self, returns: np.ndarray) -> float:
        """Calculate Profit-Risk Ratio."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
    def _calculate_average_holding_period(self) -> float:
        """Calculate Average Holding Period."""
        if len(self.prices) == 0:
            return 0.0
        return len(self.prices) / len(np.where(np.diff(self.y_pred) != 0)[0]) if len(np.where(np.diff(self.y_pred) != 0)[0]) > 0 else 0
        
    def _calculate_turnover_ratio(self) -> float:
        """Calculate Turnover Ratio."""
        if len(self.prices) == 0:
            return 0.0
        return len(np.where(np.diff(self.y_pred) != 0)[0]) / len(self.prices)
        
    def _calculate_market_timing_score(self) -> float:
        """Calculate Market Timing Score."""
        if len(self.returns) == 0 or self.benchmark_returns is None:
            return 0.0
        correct_timing = np.sum((self.returns > 0) & (self.benchmark_returns > 0)) + \
                        np.sum((self.returns < 0) & (self.benchmark_returns < 0))
        return correct_timing / len(self.returns)
        
    def _calculate_volatility_score(self, returns: np.ndarray) -> float:
        """Calculate Volatility Score."""
        if len(returns) == 0:
            return 0.0
        return 1 / (1 + np.std(returns))
        
    def _calculate_liquidity_score(self) -> float:
        """Calculate Liquidity Score."""
        if len(self.prices) == 0:
            return 0.0
        volume = np.diff(self.prices)
        return np.mean(volume) / np.std(volume) if np.std(volume) > 0 else 0
        
    def _calculate_momentum_score(self, returns: np.ndarray) -> float:
        """Calculate Momentum Score."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
    def _calculate_quality_score(self, returns: np.ndarray) -> float:
        """Calculate Quality Score."""
        if len(returns) == 0:
            return 0.0
        return (self._calculate_sharpe_ratio(returns) + 
                self._calculate_sortino_ratio(returns)) / 2
        
    def _calculate_value_score(self, returns: np.ndarray) -> float:
        """Calculate Value Score."""
        if len(returns) == 0:
            return 0.0
        return np.mean(self.prices) / np.std(self.prices) if np.std(self.prices) > 0 else 0
        
    def _calculate_growth_score(self, returns: np.ndarray) -> float:
        """Calculate Growth Score."""
        if len(returns) == 0:
            return 0.0
        return np.mean(np.diff(self.prices)) / np.std(np.diff(self.prices)) if np.std(np.diff(self.prices)) > 0 else 0
        
    def _calculate_size_score(self) -> float:
        """Calculate Size Score."""
        if len(self.prices) == 0:
            return 0.0
        return np.log(np.mean(self.prices))
        
    def _calculate_sentiment_score(self, returns: np.ndarray) -> float:
        """Calculate Sentiment Score."""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns > 0)
        
    def _calculate_technical_score(self, returns: np.ndarray) -> float:
        """Calculate Technical Score."""
        if len(returns) == 0:
            return 0.0
        return (self._calculate_momentum_score(returns) + 
                self._calculate_volatility_score(returns)) / 2
        
    def _calculate_fundamental_score(self) -> float:
        """Calculate Fundamental Score."""
        if len(self.prices) == 0:
            return 0.0
        return (self._calculate_value_score(self.returns) + 
                self._calculate_growth_score(self.returns)) / 2
        
    def _calculate_risk_score(self, returns: np.ndarray) -> float:
        """Calculate Risk Score."""
        if len(returns) == 0:
            return 0.0
        return 1 - self._calculate_volatility_score(returns)
        
    def _calculate_performance_score(self, returns: np.ndarray) -> float:
        """Calculate Performance Score."""
        if len(returns) == 0:
            return 0.0
        return (self._calculate_sharpe_ratio(returns) + 
                self._calculate_sortino_ratio(returns) + 
                self._calculate_omega_ratio(returns)) / 3
        
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
                          prices: np.ndarray = None,
                          stock_metrics: Optional[StockMetrics] = None,
                          benchmark_returns: np.ndarray = None) -> Dict[str, Any]:
    """Main evaluation function that combines all metrics."""
    evaluator = FinancialMetrics(y_true, y_pred, returns, prices, stock_metrics, benchmark_returns)
    return evaluator.get_all_metrics()

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        returns = np.random.normal(0.001, 0.02, size=n_samples)
        prices = np.cumprod(1 + returns)
        
        # Create stock metrics
        stock_metrics = StockMetrics(
            ticker="AAPL",
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        
        # Initialize metrics
        metrics = FinancialMetrics(
            y_true=y_true,
            y_pred=y_pred,
            returns=returns,
            prices=prices,
            stock_metrics=stock_metrics
        )
        
        # Compute metrics
        classification_metrics, summary = metrics.compute_classification_metrics()
        financial_metrics = metrics.compute_financial_metrics()
        
        print("Classification Metrics:")
        print(classification_metrics)
        print("\nSummary:")
        print(summary)
        print("\nFinancial Metrics:")
        print(financial_metrics)
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise

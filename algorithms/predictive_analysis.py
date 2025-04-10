# ==========================
# predictive_analysis.py
# ==========================
# Optimized algorithmic and recursive functions for predictive analysis.

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from utils.logger import get_logger
from monitoring.metrics import record_signal_metrics

logger = get_logger("algorithms")

@dataclass
class PredictiveMetrics:
    """Metrics for predictive analysis results."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    log_loss: float
    mse: float
    mae: float

class RecursivePredictor:
    """Optimized recursive predictor with caching and monitoring."""
    
    def __init__(self, 
                 max_depth: int = 10,
                 min_samples: int = 100,
                 cache_size: int = 128):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.cache_size = cache_size
        self._setup_caching()
    
    def _setup_caching(self) -> None:
        """Setup LRU caching for recursive functions."""
        self.recursive_predict = lru_cache(maxsize=self.cache_size)(self._recursive_predict)
    
    def _recursive_predict(self,
                          data: np.ndarray,
                          depth: int,
                          weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Recursive prediction function with depth control and early stopping.
        
        Args:
            data: Input data array
            depth: Current recursion depth
            weights: Optional weights for weighted prediction
            
        Returns:
            Tuple of (predictions, confidence)
        """
        if depth >= self.max_depth or len(data) < self.min_samples:
            return self._base_prediction(data, weights)
        
        # Split data and make recursive predictions
        left_data, right_data = self._split_data(data)
        left_pred, left_conf = self.recursive_predict(left_data, depth + 1, weights)
        right_pred, right_conf = self.recursive_predict(right_data, depth + 1, weights)
        
        # Combine predictions with confidence weighting
        combined_pred = (left_pred * left_conf + right_pred * right_conf) / (left_conf + right_conf)
        combined_conf = (left_conf + right_conf) / 2
        
        return combined_pred, combined_conf
    
    def _base_prediction(self,
                        data: np.ndarray,
                        weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Base prediction for leaf nodes."""
        if weights is not None:
            return np.average(data, weights=weights), 1.0
        return np.mean(data), 1.0
    
    def _split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into two parts based on median."""
        median = np.median(data)
        left_mask = data <= median
        right_mask = ~left_mask
        return data[left_mask], data[right_mask]

class LogarithmicAnalyzer:
    """Optimized logarithmic analysis with signal processing."""
    
    def __init__(self,
                 window_size: int = 20,
                 smoothing_factor: float = 0.1):
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
    
    def analyze_series(self,
                      series: pd.Series,
                      use_log: bool = True) -> Dict[str, Union[float, np.ndarray]]:
        """
        Analyze time series with logarithmic transformations.
        
        Args:
            series: Input time series
            use_log: Whether to apply logarithmic transformation
            
        Returns:
            Dictionary of analysis results
        """
        # Apply logarithmic transformation if requested
        if use_log:
            series = np.log1p(series)
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=self.window_size).mean()
        rolling_std = series.rolling(window=self.window_size).std()
        
        # Calculate momentum and acceleration
        momentum = series.diff()
        acceleration = momentum.diff()
        
        # Apply exponential smoothing
        smoothed_series = self._exponential_smoothing(series)
        
        # Calculate trend components
        trend = self._extract_trend(smoothed_series)
        seasonal = self._extract_seasonal(smoothed_series, trend)
        
        # Calculate predictive metrics
        metrics = self._calculate_metrics(series, smoothed_series)
        
        # Record metrics
        record_signal_metrics({
            'signal_type': 'logarithmic_analysis',
            'quality': metrics.accuracy,
            'generation_duration_seconds': 0.0  # TODO: Add actual timing
        })
        
        return {
            'original_series': series,
            'smoothed_series': smoothed_series,
            'trend': trend,
            'seasonal': seasonal,
            'momentum': momentum,
            'acceleration': acceleration,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'metrics': metrics
        }
    
    def _exponential_smoothing(self, series: pd.Series) -> pd.Series:
        """Apply exponential smoothing to the series."""
        smoothed = series.copy()
        for i in range(1, len(series)):
            smoothed.iloc[i] = (self.smoothing_factor * series.iloc[i] +
                              (1 - self.smoothing_factor) * smoothed.iloc[i-1])
        return smoothed
    
    def _extract_trend(self, series: pd.Series) -> pd.Series:
        """Extract trend component using moving average."""
        return series.rolling(window=self.window_size, center=True).mean()
    
    def _extract_seasonal(self,
                         series: pd.Series,
                         trend: pd.Series) -> pd.Series:
        """Extract seasonal component."""
        return series - trend
    
    def _calculate_metrics(self,
                          original: pd.Series,
                          predicted: pd.Series) -> PredictiveMetrics:
        """Calculate predictive metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            log_loss,
            mean_squared_error,
            mean_absolute_error
        )
        
        # Convert to binary classification for some metrics
        original_binary = (original > original.mean()).astype(int)
        predicted_binary = (predicted > predicted.mean()).astype(int)
        
        return PredictiveMetrics(
            accuracy=accuracy_score(original_binary, predicted_binary),
            precision=precision_score(original_binary, predicted_binary),
            recall=recall_score(original_binary, predicted_binary),
            f1_score=f1_score(original_binary, predicted_binary),
            log_loss=log_loss(original_binary, predicted),
            mse=mean_squared_error(original, predicted),
            mae=mean_absolute_error(original, predicted)
        )

class PredictivePipeline:
    """Complete predictive analysis pipeline."""
    
    def __init__(self,
                 recursive_depth: int = 10,
                 window_size: int = 20,
                 smoothing_factor: float = 0.1):
        self.recursive_predictor = RecursivePredictor(max_depth=recursive_depth)
        self.log_analyzer = LogarithmicAnalyzer(
            window_size=window_size,
            smoothing_factor=smoothing_factor
        )
    
    def analyze(self,
                data: Union[pd.Series, np.ndarray],
                use_log: bool = True) -> Dict[str, Any]:
        """
        Run complete predictive analysis pipeline.
        
        Args:
            data: Input data series or array
            use_log: Whether to use logarithmic transformation
            
        Returns:
            Dictionary of analysis results
        """
        logger.info("starting_predictive_analysis")
        
        # Convert to pandas Series if needed
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        # Run logarithmic analysis
        analysis_results = self.log_analyzer.analyze_series(data, use_log)
        
        # Run recursive prediction
        predictions, confidence = self.recursive_predictor.recursive_predict(
            data.values,
            depth=0
        )
        
        # Combine results
        results = {
            'analysis': analysis_results,
            'predictions': predictions,
            'confidence': confidence,
            'metrics': analysis_results['metrics']
        }
        
        logger.info("predictive_analysis_completed",
                   accuracy=results['metrics'].accuracy,
                   confidence=confidence)
        
        return results 
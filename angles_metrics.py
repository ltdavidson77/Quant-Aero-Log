# ==========================
# angles_metrics.py
# ==========================
# Advanced angular metrics computation with GPU acceleration and parallel processing.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numba
from numba import cuda
import cupy as cp
from scipy import signal, fft
from scipy.stats import circmean, circstd, skew, kurtosis
import warnings
import ray
from dask import dataframe as dd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from utils.logger import get_logger, log_error, log_metric, timed
from utils.config import get_config
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

logger = get_logger(__name__)

@dataclass
class AngularMetrics:
    """Enhanced container for angular metrics with validation and GPU support."""
    theta: Union[np.ndarray, cp.ndarray]  # Raw angle of price movement
    phi: Union[np.ndarray, cp.ndarray]    # Angle of smoothed price
    acc: Union[np.ndarray, cp.ndarray]    # Angular acceleration
    vol: Union[np.ndarray, cp.ndarray]    # Rolling volatility
    omega: Union[np.ndarray, cp.ndarray]  # Angular velocity
    curvature: Union[np.ndarray, cp.ndarray]  # Path curvature
    jerk: Union[np.ndarray, cp.ndarray]   # Rate of change of acceleration
    momentum: Union[np.ndarray, cp.ndarray]  # Angular momentum
    harmonics: Union[np.ndarray, cp.ndarray]  # Harmonic components
    wavelet_coeffs: Dict[str, Union[np.ndarray, cp.ndarray]]  # Wavelet coefficients
    fractal_dim: Union[np.ndarray, cp.ndarray]  # Fractal dimension
    entropy: Union[np.ndarray, cp.ndarray]  # Entropy measures
    chaos: Union[np.ndarray, cp.ndarray]  # Chaos indicators
    neural_features: Dict[str, Union[np.ndarray, cp.ndarray]]  # Neural network features
    
    def validate(self) -> None:
        """Validate metric arrays with enhanced checks."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if not isinstance(value, (np.ndarray, cp.ndarray, dict)):
                raise TypeError(f"{field} must be a numpy array, cupy array, or dict")
            if isinstance(value, (np.ndarray, cp.ndarray)):
                if np.any(np.isnan(value)):
                    warnings.warn(f"NaN values detected in {field}")
                if np.any(np.isinf(value)):
                    warnings.warn(f"Inf values detected in {field}")
    
    def to_gpu(self) -> 'AngularMetrics':
        """Convert metrics to GPU arrays."""
        gpu_metrics = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                gpu_metrics[field] = cp.asarray(value)
            elif isinstance(value, dict):
                gpu_metrics[field] = {k: cp.asarray(v) for k, v in value.items()}
            else:
                gpu_metrics[field] = value
        return AngularMetrics(**gpu_metrics)
    
    def to_cpu(self) -> 'AngularMetrics':
        """Convert metrics to CPU arrays."""
        cpu_metrics = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, cp.ndarray):
                cpu_metrics[field] = cp.asnumpy(value)
            elif isinstance(value, dict):
                cpu_metrics[field] = {k: cp.asnumpy(v) for k, v in value.items()}
            else:
                cpu_metrics[field] = value
        return AngularMetrics(**cpu_metrics)

class AngularMetricsConfig:
    """Enhanced configuration for angular metrics computation."""
    def __init__(self, 
                 window: int = 10,
                 smoothing_window: int = 5,
                 min_periods: int = 3,
                 use_gpu: bool = False,
                 normalize: bool = True,
                 num_threads: int = mp.cpu_count(),
                 wavelet_levels: int = 5,
                 pca_components: int = 3,
                 neural_layers: List[int] = [32, 16, 8]):
        self.window = window
        self.smoothing_window = smoothing_window
        self.min_periods = min_periods
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.num_threads = num_threads
        self.wavelet_levels = wavelet_levels
        self.pca_components = pca_components
        self.neural_layers = neural_layers

@numba.jit(nopython=True, parallel=True)
def _compute_angles_numba(prices: np.ndarray, 
                         window: int,
                         smoothing_window: int) -> Tuple[np.ndarray, ...]:
    """Numba-accelerated angle computation with parallel processing."""
    n = len(prices)
    theta = np.zeros(n)
    phi = np.zeros(n)
    acc = np.zeros(n)
    vol = np.zeros(n)
    omega = np.zeros(n)
    curvature = np.zeros(n)
    jerk = np.zeros(n)
    momentum = np.zeros(n)
    harmonics = np.zeros((n, 5))  # 5 harmonic components
    fractal_dim = np.zeros(n)
    entropy = np.zeros(n)
    chaos = np.zeros(n)
    
    # Compute smoothed prices in parallel
    ma = np.zeros(n)
    for i in numba.prange(n):
        start = max(0, i - smoothing_window + 1)
        ma[i] = np.mean(prices[start:i+1])
    
    # Compute metrics in parallel
    for i in numba.prange(1, n):
        # Basic angles
        theta[i] = np.arctan2(prices[i] - prices[i-1], 1)
        phi[i] = np.arctan2(ma[i] - ma[i-1], 1)
        
        # Higher order derivatives
        if i > 1:
            acc[i] = theta[i] - theta[i-1]
            omega[i] = np.sqrt(acc[i]**2 + (prices[i] - prices[i-1])**2)
            jerk[i] = acc[i] - acc[i-1]
            
            # Enhanced metrics
            curvature[i] = np.abs(jerk[i]) / (1 + omega[i]**2)**1.5
            momentum[i] = omega[i] * (prices[i] - prices[i-1])
            
            # Harmonic analysis
            for h in range(5):
                harmonics[i, h] = np.sin((h+1) * theta[i])
            
            # Fractal dimension (Hurst exponent)
            if i >= window:
                lags = np.arange(1, window)
                tau = [np.std(prices[i-l:i+1]) for l in lags]
                H = np.polyfit(np.log(lags), np.log(tau), 1)[0]
                fractal_dim[i] = 2 - H
            
            # Entropy and chaos measures
            if i >= window:
                # Sample entropy
                r = 0.2 * np.std(prices[i-window:i+1])
                m = 2
                patterns = np.array([prices[i-window+k:i-window+k+m] for k in range(window-m+1)])
                matches = np.sum([np.all(np.abs(p - q) < r) for p in patterns for q in patterns])
                entropy[i] = -np.log(matches / (window-m+1)**2)
                
                # Lyapunov exponent (chaos measure)
                divergence = np.mean(np.abs(np.diff(prices[i-window:i+1])))
                chaos[i] = np.log(divergence) / window
    
    # Compute volatility in parallel
    for i in numba.prange(window, n):
        vol[i] = np.std(prices[i-window:i+1])
    
    return (theta, phi, acc, vol, omega, curvature, jerk, momentum, 
            harmonics, fractal_dim, entropy, chaos)

@cuda.jit
def _compute_angles_gpu(prices, window, smoothing_window, 
                       theta, phi, acc, vol, omega, curvature, 
                       jerk, momentum, harmonics, fractal_dim, 
                       entropy, chaos):
    """GPU-accelerated angle computation."""
    i = cuda.grid(1)
    n = len(prices)
    
    if i < n:
        # Basic angles
        if i > 0:
            theta[i] = np.arctan2(prices[i] - prices[i-1], 1)
            
            # Compute moving average
            start = max(0, i - smoothing_window + 1)
            ma = np.mean(prices[start:i+1])
            phi[i] = np.arctan2(ma - prices[i-1], 1)
            
            # Higher order derivatives
            if i > 1:
                acc[i] = theta[i] - theta[i-1]
                omega[i] = np.sqrt(acc[i]**2 + (prices[i] - prices[i-1])**2)
                jerk[i] = acc[i] - acc[i-1]
                
                # Enhanced metrics
                curvature[i] = np.abs(jerk[i]) / (1 + omega[i]**2)**1.5
                momentum[i] = omega[i] * (prices[i] - prices[i-1])
                
                # Harmonic analysis
                for h in range(5):
                    harmonics[i, h] = np.sin((h+1) * theta[i])
        
        # Compute volatility
        if i >= window:
            vol[i] = np.std(prices[i-window:i+1])
            
            # Fractal dimension
            lags = np.arange(1, window)
            tau = np.array([np.std(prices[i-l:i+1]) for l in lags])
            H = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            fractal_dim[i] = 2 - H
            
            # Entropy and chaos measures
            r = 0.2 * np.std(prices[i-window:i+1])
            m = 2
            patterns = np.array([prices[i-window+k:i-window+k+m] for k in range(window-m+1)])
            matches = np.sum([np.all(np.abs(p - q) < r) for p in patterns for q in patterns])
            entropy[i] = -np.log(matches / (window-m+1)**2)
            
            divergence = np.mean(np.abs(np.diff(prices[i-window:i+1])))
            chaos[i] = np.log(divergence) / window

def compute_wavelet_features(data: np.ndarray, levels: int = 5) -> Dict[str, np.ndarray]:
    """Compute wavelet transform features."""
    coeffs = {}
    for level in range(1, levels + 1):
        # Compute wavelet coefficients using different wavelets
        for wavelet in ['db1', 'sym2', 'coif1']:
            cA, cD = signal.wavedec(data, wavelet, level=level)
            coeffs[f'{wavelet}_level{level}_approx'] = cA
            coeffs[f'{wavelet}_level{level}_detail'] = cD
    return coeffs

def compute_neural_features(data: np.ndarray, layers: List[int]) -> Dict[str, np.ndarray]:
    """Compute neural network-based features."""
    # Convert to PyTorch tensor
    x = torch.FloatTensor(data).unsqueeze(0)
    
    features = {}
    for i, layer_size in enumerate(layers):
        # Apply linear transformation
        linear = torch.nn.Linear(x.size(-1), layer_size)
        x = F.relu(linear(x))
        features[f'neural_layer_{i}'] = x.squeeze(0).detach().numpy()
    
    return features

@timed(logger)
def compute_angles(df: pd.DataFrame, 
                  column: str = 'price',
                  config: Optional[AngularMetricsConfig] = None) -> AngularMetrics:
    """
    Computes comprehensive angular features with enhanced metrics and GPU support.
    
    Args:
        df: Input DataFrame
        column: Column name for price data
        config: Configuration for metric computation
        
    Returns:
        AngularMetrics object containing all computed metrics
    """
    try:
        if config is None:
            config = AngularMetricsConfig()
            
        # Input validation
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
            
        prices = df[column].values
        if len(prices) < config.min_periods:
            raise ValueError("Insufficient data points for computation")
            
        if config.use_gpu and cuda.is_available():
            # GPU computation
            prices_gpu = cp.asarray(prices)
            n = len(prices)
            
            # Allocate GPU memory
            theta_gpu = cp.zeros(n)
            phi_gpu = cp.zeros(n)
            acc_gpu = cp.zeros(n)
            vol_gpu = cp.zeros(n)
            omega_gpu = cp.zeros(n)
            curvature_gpu = cp.zeros(n)
            jerk_gpu = cp.zeros(n)
            momentum_gpu = cp.zeros(n)
            harmonics_gpu = cp.zeros((n, 5))
            fractal_dim_gpu = cp.zeros(n)
            entropy_gpu = cp.zeros(n)
            chaos_gpu = cp.zeros(n)
            
            # Launch GPU kernel
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            _compute_angles_gpu[blocks_per_grid, threads_per_block](
                prices_gpu, config.window, config.smoothing_window,
                theta_gpu, phi_gpu, acc_gpu, vol_gpu, omega_gpu,
                curvature_gpu, jerk_gpu, momentum_gpu, harmonics_gpu,
                fractal_dim_gpu, entropy_gpu, chaos_gpu
            )
            
            # Get results
            metrics = (theta_gpu, phi_gpu, acc_gpu, vol_gpu, omega_gpu,
                      curvature_gpu, jerk_gpu, momentum_gpu, harmonics_gpu,
                      fractal_dim_gpu, entropy_gpu, chaos_gpu)
        else:
            # CPU computation with parallel processing
            metrics = _compute_angles_numba(
                prices,
                config.window,
                config.smoothing_window
            )
        
        # Compute additional features
        wavelet_coeffs = compute_wavelet_features(prices, config.wavelet_levels)
        neural_features = compute_neural_features(prices, config.neural_layers)
        
        # Create AngularMetrics object
        result = AngularMetrics(
            *metrics,
            wavelet_coeffs=wavelet_coeffs,
            neural_features=neural_features
        )
        result.validate()
        
        return result
        
    except Exception as e:
        log_error(e, {"function": "compute_angles"})
        raise

@timed(logger)
def compute_angle_dataframe(df: pd.DataFrame,
                          column: str = 'price',
                          config: Optional[AngularMetricsConfig] = None) -> pd.DataFrame:
    """
    Computes angular metrics and returns as DataFrame with enhanced features.
    
    Args:
        df: Input DataFrame
        column: Column name for price data
        config: Configuration for metric computation
        
    Returns:
        DataFrame containing all angular metrics
    """
    try:
        metrics = compute_angles(df, column, config)
        
        # Create DataFrame with basic metrics
        result = pd.DataFrame({
            'theta': metrics.theta,
            'phi': metrics.phi,
            'acc': metrics.acc,
            'vol': metrics.vol,
            'omega': metrics.omega,
            'curvature': metrics.curvature,
            'jerk': metrics.jerk,
            'momentum': metrics.momentum
        }, index=df.index)
        
        # Add harmonic components
        for i in range(5):
            result[f'harmonic_{i+1}'] = metrics.harmonics[:, i]
        
        # Add wavelet features
        for name, coeffs in metrics.wavelet_coeffs.items():
            result[f'wavelet_{name}'] = coeffs
        
        # Add neural features
        for name, features in metrics.neural_features.items():
            result[f'neural_{name}'] = features
        
        # Add derived features
        result['theta_ma'] = result['theta'].rolling(
            window=config.smoothing_window,
            min_periods=config.min_periods
        ).mean()
        
        result['phi_ma'] = result['phi'].rolling(
            window=config.smoothing_window,
            min_periods=config.min_periods
        ).mean()
        
        # Add circular statistics
        result['theta_circmean'] = result['theta'].rolling(
            window=config.window,
            min_periods=config.min_periods
        ).apply(circmean)
        
        result['theta_circstd'] = result['theta'].rolling(
            window=config.window,
            min_periods=config.min_periods
        ).apply(circstd)
        
        # Add signal processing features
        result['theta_fft'] = np.abs(fft.fft(result['theta']))
        result['phi_fft'] = np.abs(fft.fft(result['phi']))
        
        # Add trend features
        result['trend_strength'] = np.abs(result['theta'] - result['phi'])
        result['trend_consistency'] = result['theta'].rolling(
            window=config.window,
            min_periods=config.min_periods
        ).apply(lambda x: np.mean(np.sign(x) == np.sign(x[0])))
        
        # Add fractal and chaos features
        result['fractal_dim'] = metrics.fractal_dim
        result['entropy'] = metrics.entropy
        result['chaos'] = metrics.chaos
        
        # Add statistical features
        result['skewness'] = result['theta'].rolling(
            window=config.window,
            min_periods=config.min_periods
        ).apply(skew)
        
        result['kurtosis'] = result['theta'].rolling(
            window=config.window,
            min_periods=config.min_periods
        ).apply(kurtosis)
        
        # Normalize if requested
        if config.normalize:
            scaler = StandardScaler()
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        
        # Add PCA components
        pca = PCA(n_components=config.pca_components)
        pca_features = pca.fit_transform(result[numeric_cols])
        for i in range(config.pca_components):
            result[f'pca_{i+1}'] = pca_features[:, i]
        
        return result
        
    except Exception as e:
        log_error(e, {"function": "compute_angle_dataframe"})
        raise

@timed(logger)
def analyze_angular_patterns(df: pd.DataFrame,
                           threshold: float = 0.1) -> Dict[str, List[int]]:
    """
    Analyzes angular patterns in the data to identify significant movements.
    
    Args:
        df: DataFrame with angular metrics
        threshold: Threshold for significant movements
        
    Returns:
        Dictionary of pattern indices
    """
    try:
        patterns = {
            'sharp_turns': [],
            'trend_reversals': [],
            'momentum_shifts': [],
            'volatility_spikes': [],
            'harmonic_resonance': [],
            'chaos_transitions': [],
            'fractal_breaks': [],
            'neural_anomalies': []
        }
        
        # Detect sharp turns
        sharp_turns = np.where(np.abs(df['curvature']) > threshold)[0]
        patterns['sharp_turns'] = sharp_turns.tolist()
        
        # Detect trend reversals
        theta_diff = np.diff(df['theta'])
        reversals = np.where(np.abs(theta_diff) > np.pi/2)[0]
        patterns['trend_reversals'] = reversals.tolist()
        
        # Detect momentum shifts
        momentum_diff = np.diff(df['momentum'])
        shifts = np.where(np.abs(momentum_diff) > threshold)[0]
        patterns['momentum_shifts'] = shifts.tolist()
        
        # Detect volatility spikes
        vol_mean = df['vol'].mean()
        vol_std = df['vol'].std()
        spikes = np.where(df['vol'] > vol_mean + 2*vol_std)[0]
        patterns['volatility_spikes'] = spikes.tolist()
        
        # Detect harmonic resonance
        harmonic_sum = np.sum([df[f'harmonic_{i+1}'] for i in range(5)], axis=0)
        resonance = np.where(harmonic_sum > threshold)[0]
        patterns['harmonic_resonance'] = resonance.tolist()
        
        # Detect chaos transitions
        chaos_diff = np.diff(df['chaos'])
        transitions = np.where(np.abs(chaos_diff) > threshold)[0]
        patterns['chaos_transitions'] = transitions.tolist()
        
        # Detect fractal breaks
        fractal_diff = np.diff(df['fractal_dim'])
        breaks = np.where(np.abs(fractal_diff) > threshold)[0]
        patterns['fractal_breaks'] = breaks.tolist()
        
        # Detect neural anomalies
        neural_cols = [col for col in df.columns if col.startswith('neural_')]
        neural_data = df[neural_cols].values
        neural_mean = np.mean(neural_data, axis=0)
        neural_std = np.std(neural_data, axis=0)
        anomalies = np.where(np.any(np.abs(neural_data - neural_mean) > 3*neural_std, axis=1))[0]
        patterns['neural_anomalies'] = anomalies.tolist()
        
        return patterns
        
    except Exception as e:
        log_error(e, {"function": "analyze_angular_patterns"})
        raise

@timed(logger)
def print_angular_summary(df: pd.DataFrame) -> None:
    """
    Prints comprehensive summary of angular metrics with enhanced statistics.
    
    Args:
        df: DataFrame with angular metrics
    """
    try:
        logger.info("Angular Metric Summary:")
        
        # Basic statistics
        summary = df.describe()
        logger.info("Basic Statistics:\n%s", summary)
        
        # Correlation analysis
        corr = df.corr()
        logger.info("Correlation Matrix:\n%s", corr)
        
        # Pattern analysis
        patterns = analyze_angular_patterns(df)
        logger.info("Detected Patterns:")
        for pattern, indices in patterns.items():
            logger.info("%s: %d occurrences", pattern, len(indices))
            
        # Trend analysis
        trend_strength = df['trend_strength'].mean()
        trend_consistency = df['trend_consistency'].mean()
        logger.info("Trend Analysis:")
        logger.info("Average Trend Strength: %.4f", trend_strength)
        logger.info("Trend Consistency: %.4f", trend_consistency)
        
        # Chaos and fractal analysis
        chaos_mean = df['chaos'].mean()
        fractal_mean = df['fractal_dim'].mean()
        logger.info("Chaos and Fractal Analysis:")
        logger.info("Average Chaos Level: %.4f", chaos_mean)
        logger.info("Average Fractal Dimension: %.4f", fractal_mean)
        
        # Neural feature analysis
        neural_cols = [col for col in df.columns if col.startswith('neural_')]
        if neural_cols:
            neural_summary = df[neural_cols].describe()
            logger.info("Neural Feature Summary:\n%s", neural_summary)
        
    except Exception as e:
        log_error(e, {"function": "print_angular_summary"})
        raise

# ==========================
#   log_signal.py 
# ==========================
# Constructs multi-layered logarithmic signals using angular input metrics with quantum enhancements.

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from functools import lru_cache
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal
from scipy.fft import fft, ifft
from scipy.stats import entropy
import numba
from monitoring.metrics import record_signal_metrics, SignalQualityMetrics
from quantum.quantum_state import QuantumState, QuantumStateManager, QuantumOptimizationConfig
from utils.cache import CacheManager, CacheConfig
from utils.validation import validate_input_data
from utils.performance import PerformanceMonitor
from config.signal_config import SignalConfig, SignalType

logger = logging.getLogger(__name__)

class SignalProcessingMode(Enum):
    """Different modes of signal processing."""
    STANDARD = "standard"
    QUANTUM_ENHANCED = "quantum_enhanced"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class SignalMetrics:
    """Comprehensive metrics for signal generation results."""
    quality: float
    stability: float
    entropy: float
    generation_time: float
    cache_hit: bool
    quantum_enhancement_factor: float
    noise_level: float
    frequency_components: Dict[str, float]
    correlation_strength: float
    prediction_accuracy: float
    confidence_interval: Tuple[float, float]

class LogSignalGenerator:
    """Enhanced logarithmic signal generator with quantum optimizations and advanced processing."""
    
    def __init__(self,
                 config: SignalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quantum_config: Optional[QuantumOptimizationConfig] = None):
        """
        Initialize signal generator with comprehensive configuration.
        
        Args:
            config: Signal configuration parameters
            cache_config: Optional cache configuration
            quantum_config: Optional quantum optimization configuration
        """
        self.config = config
        self.cache = CacheManager(cache_config or CacheConfig())
        self.quantum_state_manager = QuantumStateManager(quantum_config)
        self.performance_monitor = PerformanceMonitor()
        self.signal_history = []
        self.metrics_history = []
        
    @numba.jit(nopython=True)
    def _compute_base_signal(self,
                           theta: np.ndarray,
                           phi: np.ndarray,
                           acc: np.ndarray,
                           vol: np.ndarray) -> np.ndarray:
        """
        Compute base logarithmic signal with Numba optimization.
        
        Args:
            theta: Angular momentum values
            phi: Direction values
            acc: Acceleration values
            vol: Volatility values
            
        Returns:
            Base signal array
        """
        try:
            # Compute signal components with adaptive weights
            weights = self._compute_adaptive_weights(theta, phi, acc, vol)
            
            L1 = np.log(1 + weights['alpha'] * np.sin(theta)**2 - weights['beta'] * np.cos(phi)**2)
            L2 = np.log(1 + weights['gamma'] * vol**2 * np.log(1 + np.abs(np.sin(theta - phi))))
            L3 = np.log(1 + weights['delta'] * np.abs(acc) * np.sin(theta)**2)
            
            # Combine components with non-linear transformation
            signal = self._apply_nonlinear_transformation(L1 + L2 + L3)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error computing base signal: {str(e)}")
            raise
            
    def _compute_adaptive_weights(self,
                                theta: np.ndarray,
                                phi: np.ndarray,
                                acc: np.ndarray,
                                vol: np.ndarray) -> Dict[str, float]:
        """Compute adaptive weights based on input characteristics."""
        # Calculate volatility-based adjustment
        vol_adjustment = np.mean(vol) / np.std(vol)
        
        # Calculate momentum-based adjustment
        momentum = np.mean(np.abs(np.diff(theta)))
        momentum_adjustment = 1 / (1 + momentum)
        
        # Calculate direction-based adjustment
        direction_change = np.mean(np.abs(np.diff(phi)))
        direction_adjustment = 1 / (1 + direction_change)
        
        return {
            'alpha': self.config.alpha * momentum_adjustment,
            'beta': self.config.beta * direction_adjustment,
            'gamma': self.config.gamma * vol_adjustment,
            'delta': self.config.delta * (momentum_adjustment + direction_adjustment) / 2
        }
        
    def _apply_nonlinear_transformation(self, signal: np.ndarray) -> np.ndarray:
        """Apply non-linear transformation to enhance signal characteristics."""
        # Apply sigmoid transformation
        transformed = 1 / (1 + np.exp(-signal))
        
        # Apply frequency domain filtering
        freq_domain = fft(transformed)
        filtered = self._apply_frequency_filter(freq_domain)
        
        # Apply quantum enhancement if enabled
        if self.config.processing_mode == SignalProcessingMode.QUANTUM_ENHANCED:
            filtered = self._apply_quantum_enhancement(filtered)
            
        return ifft(filtered).real
        
    def _apply_frequency_filter(self, freq_domain: np.ndarray) -> np.ndarray:
        """Apply adaptive frequency filtering."""
        # Calculate power spectrum
        power = np.abs(freq_domain)**2
        
        # Identify dominant frequencies
        threshold = np.mean(power) + 2 * np.std(power)
        mask = power > threshold
        
        # Apply filter
        filtered = freq_domain * mask
        
        return filtered
        
    def _apply_quantum_enhancement(self, signal: np.ndarray) -> np.ndarray:
        """Apply quantum enhancement to the signal."""
        quantum_state = self.quantum_state_manager.get_state('signal_optimization')
        if quantum_state is not None:
            # Prepare quantum state
            quantum_state.prepare(signal)
            
            # Apply quantum operations
            quantum_state.apply_quantum_gates()
            
            # Measure and return enhanced signal
            return quantum_state.measure()
        return signal
        
    def compute_signal(self,
                      theta: np.ndarray,
                      phi: np.ndarray,
                      acc: np.ndarray,
                      vol: np.ndarray) -> Tuple[np.ndarray, SignalMetrics]:
        """
        Compute enhanced logarithmic signal with comprehensive metrics.
        
        Args:
            theta: Angular momentum values
            phi: Direction values
            acc: Acceleration values
            vol: Volatility values
            
        Returns:
            Tuple of (signal array, signal metrics)
        """
        with self.performance_monitor.measure():
            try:
                # Validate inputs
                validate_input_data(theta, phi, acc, vol)
                
                # Check cache
                cache_key = self._generate_cache_key(theta, phi, acc, vol)
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    return self._get_cached_metrics(cached_result)
                
                # Compute base signal
                base_signal = self._compute_base_signal(theta, phi, acc, vol)
                
                # Apply post-processing
                processed_signal = self._apply_post_processing(base_signal)
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(processed_signal)
                
                # Update history
                self._update_history(processed_signal, metrics)
                
                # Cache result
                self.cache.set(cache_key, (processed_signal, metrics))
                
                # Record metrics
                self._record_metrics(metrics)
                
                return processed_signal, metrics
                
            except Exception as e:
                logger.error(f"Error computing signal: {str(e)}")
                raise
                
    def _apply_post_processing(self, signal: np.ndarray) -> np.ndarray:
        """Apply post-processing to enhance signal quality."""
        # Apply smoothing
        smoothed = signal.savgol_filter(signal, window_length=5, polyorder=2)
        
        # Apply adaptive thresholding
        threshold = np.mean(smoothed) + 2 * np.std(smoothed)
        thresholded = np.where(smoothed > threshold, smoothed, 0)
        
        # Apply normalization
        normalized = (thresholded - np.min(thresholded)) / (np.max(thresholded) - np.min(thresholded))
        
        return normalized
        
    def _calculate_comprehensive_metrics(self, signal: np.ndarray) -> SignalMetrics:
        """Calculate comprehensive signal metrics."""
        # Basic metrics
        stability = 1 - np.std(signal) / np.mean(np.abs(signal))
        hist, _ = np.histogram(signal, bins=100, density=True)
        signal_entropy = entropy(hist)
        
        # Frequency analysis
        freq_components = self._analyze_frequency_components(signal)
        
        # Correlation analysis
        correlation = self._analyze_correlation(signal)
        
        # Prediction accuracy
        accuracy = self._calculate_prediction_accuracy(signal)
        
        # Confidence interval
        ci = self._calculate_confidence_interval(signal)
        
        # Quantum enhancement factor
        q_factor = self._calculate_quantum_enhancement_factor(signal)
        
        # Noise level
        noise = self._calculate_noise_level(signal)
        
        return SignalMetrics(
            quality=float(stability * (1 - signal_entropy) * accuracy),
            stability=float(stability),
            entropy=float(signal_entropy),
            generation_time=float(self.performance_monitor.last_duration),
            cache_hit=False,
            quantum_enhancement_factor=float(q_factor),
            noise_level=float(noise),
            frequency_components=freq_components,
            correlation_strength=float(correlation),
            prediction_accuracy=float(accuracy),
            confidence_interval=ci
        )
        
    def _analyze_frequency_components(self, signal: np.ndarray) -> Dict[str, float]:
        """Analyze frequency components of the signal."""
        freq_domain = fft(signal)
        power = np.abs(freq_domain)**2
        
        return {
            'low_frequency': float(np.mean(power[:len(power)//4])),
            'medium_frequency': float(np.mean(power[len(power)//4:len(power)//2])),
            'high_frequency': float(np.mean(power[len(power)//2:]))
        }
        
    def _analyze_correlation(self, signal: np.ndarray) -> float:
        """Analyze correlation with historical signals."""
        if not self.signal_history:
            return 0.0
            
        correlations = [np.corrcoef(signal, hist)[0,1] for hist in self.signal_history]
        return float(np.mean(correlations))
        
    def _calculate_prediction_accuracy(self, signal: np.ndarray) -> float:
        """Calculate prediction accuracy based on historical data."""
        if len(self.metrics_history) < 2:
            return 0.5
            
        # Use last prediction accuracy as baseline
        return self.metrics_history[-1].prediction_accuracy
        
    def _calculate_confidence_interval(self, signal: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for the signal."""
        mean = np.mean(signal)
        std = np.std(signal)
        return (float(mean - 2*std), float(mean + 2*std))
        
    def _calculate_quantum_enhancement_factor(self, signal: np.ndarray) -> float:
        """Calculate quantum enhancement factor."""
        if not self.quantum_state_manager.is_enabled():
            return 1.0
            
        # Compare with classical version
        classical_signal = self._compute_base_signal(signal, signal, signal, signal)
        return float(np.mean(np.abs(signal - classical_signal)))
        
    def _calculate_noise_level(self, signal: np.ndarray) -> float:
        """Calculate noise level in the signal."""
        # Apply high-pass filter to isolate noise
        b, a = signal.butter(4, 0.1, 'high')
        noise = signal.filtfilt(b, a, signal)
        return float(np.std(noise))
        
    def _update_history(self, signal: np.ndarray, metrics: SignalMetrics) -> None:
        """Update signal and metrics history."""
        self.signal_history.append(signal)
        self.metrics_history.append(metrics)
        
        # Keep history size limited
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
            self.metrics_history.pop(0)
            
    def _record_metrics(self, metrics: SignalMetrics) -> None:
        """Record metrics to monitoring system."""
        record_signal_metrics({
            'signal_type': self.config.signal_type.value,
            'quality': metrics.quality,
            'stability': metrics.stability,
            'entropy': metrics.entropy,
            'generation_duration_seconds': metrics.generation_time,
            'quantum_enhancement_factor': metrics.quantum_enhancement_factor,
            'noise_level': metrics.noise_level,
            'correlation_strength': metrics.correlation_strength,
            'prediction_accuracy': metrics.prediction_accuracy
        })

# ----------------------------------------
# Batch Construction from DataFrame Input
# ----------------------------------------
def build_log_signal_from_df(df: pd.DataFrame,
                           config: Optional[SignalConfig] = None) -> Tuple[pd.Series, SignalMetrics]:
    """
    Construct log-based signal from DataFrame input with comprehensive configuration.
    
    Args:
        df: DataFrame containing input metrics
        config: Optional signal configuration
        
    Returns:
        Tuple of (signal Series, signal metrics)
    """
    config = config or SignalConfig()
    generator = LogSignalGenerator(config)
    
    # Validate DataFrame columns
    required_columns = ['theta', 'phi', 'acc', 'vol']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
    signal, metrics = generator.compute_signal(
        df['theta'].values,
        df['phi'].values,
        df['acc'].values,
        df['vol'].values
    )
    
    return pd.Series(signal, index=df.index), metrics

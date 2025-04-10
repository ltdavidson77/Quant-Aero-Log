# ==========================
# alg_signal.py
# ==========================
# Enhanced algorithmic signal generator with comprehensive features.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
from scipy import signal
from scipy.stats import zscore, kurtosis, skew, normaltest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pywt
import numba
from prometheus_client import Counter, Gauge, Histogram, Summary
import psutil
import gc
import cupy as cp
import json
import os
from pathlib import Path
import hashlib
import uuid
from functools import partial
import inspect
import sys
import base64

# Local imports
from config_env import (
    DEBUG_MODE,
    ALG_SIGNAL_VALIDATION_ENABLED,
    ALG_SIGNAL_MONITORING_ENABLED,
    ALG_SIGNAL_CACHE_ENABLED,
    ALG_SIGNAL_GPU_ENABLED,
    ALG_SIGNAL_SECURITY_ENABLED
)

logger = logging.getLogger(__name__)

# Prometheus metrics
ALG_SIGNAL_PROCESSING_COUNTER = Counter('alg_signal_processing_total', 'Total number of algorithmic signals processed')
ALG_SIGNAL_PROCESSING_ERRORS = Counter('alg_signal_processing_errors_total', 'Total number of algorithmic processing errors')
ALG_SIGNAL_PROCESSING_DURATION = Histogram('alg_signal_processing_duration_seconds', 'Time spent processing algorithmic signals')
ALG_SIGNAL_QUALITY = Gauge('alg_signal_quality', 'Quality score of algorithmic signals')
ALG_SIGNAL_MEMORY_USAGE = Gauge('alg_signal_memory_usage_bytes', 'Memory usage during algorithmic processing')
ALG_SIGNAL_GPU_MEMORY = Gauge('alg_signal_gpu_memory_bytes', 'GPU memory usage')
ALG_SIGNAL_DATA_QUALITY = Gauge('alg_signal_data_quality_score', 'Data quality score')
ALG_SIGNAL_STATIONARITY = Gauge('alg_signal_stationarity', 'Signal stationarity score')
ALG_SIGNAL_ENTROPY = Gauge('alg_signal_entropy', 'Signal entropy')
ALG_SIGNAL_OUTLIER_COUNT = Counter('alg_signal_outliers_total', 'Total number of outliers detected')

# ------------------------------------------
# Original Algorithmic Signal Construction Function
# ------------------------------------------
def compute_alg_signal(theta, phi, acc, vol, a1=0.8, b1=0.6, g1=0.4):
    """
    Computes non-logarithmic signal using:
    - a1: weight on angular interference
    - b1: weight on acceleration*direction
    - g1: weight on volatility-amplitude cross product
    """
    A1 = a1 * np.sin(theta + 0.5 * phi)**2
    A2 = b1 * (acc**2) * np.cos(phi)**2
    A3 = g1 * vol * np.abs(np.sin(phi) * np.cos(theta))
    return A1 + A2 + A3

# ------------------------------------------
# Original Batch Construction from DataFrame Input
# ------------------------------------------
def build_alg_signal_from_df(df):
    return compute_alg_signal(
        df['theta'].values,
        df['phi'].values,
        df['acc'].values,
        df['vol'].values
    )

class SignalProcessingMethod(Enum):
    """Supported signal processing methods."""
    WAVELET = 'wavelet'
    FFT = 'fft'
    SAVITZKY_GOLAY = 'savitzky_golay'
    KALMAN = 'kalman'
    HODRICK_PRESCOTT = 'hodrick_prescott'
    CUSTOM = 'custom'

class SignalType(Enum):
    """Supported signal types."""
    ANGULAR = 'angular'
    VOLATILITY = 'volatility'
    MOMENTUM = 'momentum'
    TREND = 'trend'
    OSCILLATOR = 'oscillator'
    CUSTOM = 'custom'

@dataclass
class SignalMetrics:
    """Metrics for signal analysis."""
    stationarity: float
    entropy: float
    kurtosis: float
    skewness: float
    normality: float
    autocorrelation: float
    trend_strength: float
    seasonality_strength: float
    noise_level: float
    signal_to_noise: float
    volatility: float
    momentum: float
    mean_reversion: float

@dataclass
class SignalParameters:
    """Parameters for signal computation."""
    a1: float = 0.8  # Angular interference weight
    b1: float = 0.6  # Acceleration-direction weight
    g1: float = 0.4  # Volatility-amplitude weight
    alpha: float = 0.1  # Smoothing factor
    beta: float = 0.2  # Trend factor
    gamma: float = 0.3  # Volatility factor
    delta: float = 0.4  # Momentum factor
    epsilon: float = 0.5  # Mean reversion factor
    window_size: int = 20  # Rolling window size
    threshold: float = 0.5  # Signal threshold

@numba.jit(nopython=True)
def _savitzky_golay_filter(y, window_size, order):
    """Numba-accelerated Savitzky-Golay filter."""
    try:
        window_size = np.abs(np.int64(window_size))
        order = np.abs(np.int64(order))
        
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
            
        half_window = (window_size - 1) // 2
        b = np.mat([[k**i for i in range(order+1)] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[0]
        
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        
        return np.convolve(m[::-1], y, mode='valid')
    except Exception as e:
        logger.error(f"Savitzky-Golay filter error: {str(e)}")
        return y

@numba.jit(nopython=True)
def _kalman_filter(y, process_noise=1e-4, measurement_noise=1e-1):
    """Numba-accelerated Kalman filter."""
    try:
        n_iter = len(y)
        sz = (n_iter,)
        
        # Initial state
        x = y[0]
        
        # Initial uncertainty
        P = 1.0
        
        # Process noise
        Q = process_noise
        
        # Measurement noise
        R = measurement_noise
        
        # Allocate space for arrays
        xhat = np.zeros(sz)
        Pminus = np.zeros(sz)
        K = np.zeros(sz)
        
        xhat[0] = x
        Pminus[0] = P
        
        for k in range(1, n_iter):
            # Time update
            xminus = xhat[k-1]
            Pminus[k] = Pminus[k-1] + Q
            
            # Measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xminus + K[k] * (y[k] - xminus)
            Pminus[k] = (1 - K[k]) * Pminus[k]
            
        return xhat
    except Exception as e:
        logger.error(f"Kalman filter error: {str(e)}")
        return y

@numba.jit(nopython=True)
def _hodrick_prescott_filter(y, lamb=1600):
    """Numba-accelerated Hodrick-Prescott filter."""
    try:
        n = len(y)
        I = np.eye(n)
        D = np.zeros((n-2, n))
        
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
            
        trend = np.linalg.solve(I + lamb * D.T @ D, y)
        cycle = y - trend
        
        return trend, cycle
    except Exception as e:
        logger.error(f"Hodrick-Prescott filter error: {str(e)}")
        return y, np.zeros_like(y)

@dataclass
class FrequencyAnalysis:
    """Results of frequency domain analysis."""
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    dominant_frequency: float
    bandwidth: float
    spectral_entropy: float
    spectral_flatness: float
    spectral_centroid: float
    spectral_rolloff: float

@dataclass
class CrossCorrelation:
    """Results of cross-correlation analysis."""
    correlation: np.ndarray
    lag: np.ndarray
    max_correlation: float
    max_lag: int
    phase_difference: float
    coherence: float

@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    completeness: float
    consistency: float
    accuracy: float
    timeliness: float
    validity: float
    uniqueness: float
    integrity: float
    precision: float
    recall: float
    f1_score: float

@dataclass
class AnomalyDetection:
    """Results of anomaly detection."""
    anomalies: np.ndarray
    anomaly_scores: np.ndarray
    threshold: float
    num_anomalies: int
    anomaly_indices: np.ndarray
    confidence_scores: np.ndarray

@dataclass
class Alert:
    """Alert information."""
    timestamp: float
    level: str
    message: str
    metric: str
    value: float
    threshold: float
    action: str

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: float
    user: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any]

class AlgorithmicSignalGenerator:
    """Enhanced algorithmic signal generator with comprehensive features."""
    
    def __init__(self,
                 signal_type: SignalType = SignalType.ANGULAR,
                 processing_method: SignalProcessingMethod = SignalProcessingMethod.WAVELET,
                 validation_enabled: bool = ALG_SIGNAL_VALIDATION_ENABLED,
                 monitoring_enabled: bool = ALG_SIGNAL_MONITORING_ENABLED,
                 cache_enabled: bool = ALG_SIGNAL_CACHE_ENABLED,
                 gpu_enabled: bool = ALG_SIGNAL_GPU_ENABLED,
                 security_enabled: bool = ALG_SIGNAL_SECURITY_ENABLED,
                 parallel_enabled: bool = True,
                 batch_size: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 audit_log_path: Optional[str] = None):
        """
        Initialize the algorithmic signal generator with enhanced features.
        """
        self.signal_type = signal_type
        self.processing_method = processing_method
        self.validation_enabled = validation_enabled
        self.monitoring_enabled = monitoring_enabled
        self.cache_enabled = cache_enabled
        self.gpu_enabled = gpu_enabled
        self.security_enabled = security_enabled
        self.parallel_enabled = parallel_enabled
        self.batch_size = batch_size
        
        # Initialize components
        self._setup_gpu()
        self._setup_cache()
        self._setup_metadata()
        self._setup_security()
        self._setup_parallel()
        
        # Initialize parameters
        self.params = SignalParameters()
        
        # Initialize alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'processing_time_ms': 1000.0,
            'memory_usage_gb': 4.0,
            'gpu_memory_gb': 2.0,
            'error_rate': 0.1,
            'anomaly_rate': 0.05,
            'data_quality_score': 0.8,
            'signal_quality_score': 0.7
        }
        
        # Initialize alert history
        self.alert_history = []
        
        # Initialize security components
        self.audit_log_path = audit_log_path or "audit_log.json"
        self.audit_log = []
        
    def _setup_gpu(self) -> None:
        """Setup GPU components."""
        if self.gpu_enabled:
            try:
                self.gpu_available = cp.is_available()
                if self.gpu_available:
                    self.gpu_memory = cp.cuda.Device(0).mem_info
            except Exception as e:
                logger.warning(f"GPU setup failed: {str(e)}")
                self.gpu_available = False
                
    def _setup_cache(self) -> None:
        """Setup caching components."""
        if self.cache_enabled:
            self._cache = {}
            self._cache_metadata = {}
            
    def _setup_metadata(self) -> None:
        """Setup metadata tracking."""
        self.metadata = {
            'timestamp': time.time(),
            'processing_time_ms': 0,
            'error_count': 0,
            'warning_count': 0,
            'signal_quality': 1.0,
            'performance_metrics': {},
            'validation_metrics': {},
            'signal_metrics': {},
            'data_quality_metrics': {},
            'security_metrics': {},
            'frequency_analysis': {},
            'anomalies': {}
        }
        
    def _setup_security(self) -> None:
        """Setup security components."""
        if self.security_enabled:
            self.encryption_key = self._generate_encryption_key()
            self.access_control = self._setup_access_control()
            self._load_audit_log()
            
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure storage."""
        return hashlib.sha256(str(uuid.uuid4()).encode()).digest()
        
    def _setup_access_control(self) -> Dict[str, List[str]]:
        """Setup access control rules."""
        return {
            'read': ['admin', 'user'],
            'write': ['admin'],
            'execute': ['admin', 'user']
        }
        
    def _setup_parallel(self) -> None:
        """Setup parallel processing components."""
        if self.parallel_enabled:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init()
                self.ray_available = True
            except Exception as e:
                logger.warning(f"Ray setup failed: {str(e)}")
                self.ray_available = False
                
    def _process_signal_cpu(self, data: np.ndarray) -> np.ndarray:
        """Process signal using CPU-accelerated methods."""
        try:
            if self.processing_method == SignalProcessingMethod.SAVITZKY_GOLAY:
                return _savitzky_golay_filter(data, window_size=11, order=3)
            elif self.processing_method == SignalProcessingMethod.KALMAN:
                return _kalman_filter(data)
            elif self.processing_method == SignalProcessingMethod.HODRICK_PRESCOTT:
                trend, _ = _hodrick_prescott_filter(data)
                return trend
            else:
                return data
        except Exception as e:
            logger.error(f"CPU processing error: {str(e)}")
            return data
            
    def _process_signal_gpu(self, data: np.ndarray) -> np.ndarray:
        """Process signal using GPU acceleration."""
        if not self.gpu_available:
            return self._process_signal_cpu(data)
            
        try:
            # Convert to GPU array
            gpu_data = cp.asarray(data)
            
            # Process on GPU
            if self.processing_method == SignalProcessingMethod.WAVELET:
                # Perform wavelet transform on GPU
                coeffs = pywt.wavedec(gpu_data, 'db1', level=3)
                processed = pywt.waverec(coeffs, 'db1')
            elif self.processing_method == SignalProcessingMethod.FFT:
                # Perform FFT on GPU
                fft_data = cp.fft.fft(gpu_data)
                processed = cp.fft.ifft(fft_data).real
            else:
                processed = gpu_data
                
            # Convert back to CPU
            return cp.asnumpy(processed)
            
        except Exception as e:
            logger.error(f"GPU processing error: {str(e)}")
            return self._process_signal_cpu(data)
            
    def _process_batch(self, data: np.ndarray) -> np.ndarray:
        """Process data in batches for memory efficiency."""
        try:
            n_samples = len(data)
            processed = np.zeros_like(data)
            
            for i in range(0, n_samples, self.batch_size):
                batch = data[i:i+self.batch_size]
                if self.gpu_enabled:
                    processed[i:i+self.batch_size] = self._process_signal_gpu(batch)
                else:
                    processed[i:i+self.batch_size] = self._process_signal_cpu(batch)
                    
            return processed
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return data
            
    def _process_parallel(self, data: np.ndarray) -> np.ndarray:
        """Process data in parallel using Ray."""
        if not self.ray_available:
            return self._process_batch(data)
            
        try:
            import ray
            
            @ray.remote
            def process_chunk(chunk):
                if self.gpu_enabled:
                    return self._process_signal_gpu(chunk)
                else:
                    return self._process_signal_cpu(chunk)
                    
            # Split data into chunks
            n_samples = len(data)
            chunk_size = n_samples // ray.available_resources()['CPU']
            chunks = [data[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]
            
            # Process chunks in parallel
            processed_chunks = ray.get([process_chunk.remote(chunk) for chunk in chunks])
            
            # Combine results
            return np.concatenate(processed_chunks)
        except Exception as e:
            logger.error(f"Parallel processing error: {str(e)}")
            return self._process_batch(data)
            
    def _compute_signal_metrics(self, signal: np.ndarray) -> SignalMetrics:
        """Compute comprehensive signal metrics."""
        try:
            # Stationarity test (ADF)
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(signal)
            stationarity = -adf_result[0]  # More negative = more stationary
            
            # Entropy
            hist, _ = np.histogram(signal, bins=50, density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Statistical properties
            kurt = kurtosis(signal)
            skewness = skew(signal)
            normality = normaltest(signal)[1]  # p-value
            
            # Autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr[1] / autocorr[0]  # First lag
            
            # Trend and seasonality
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(signal, period=24)
            trend_strength = np.std(decomposition.trend) / np.std(signal)
            seasonality_strength = np.std(decomposition.seasonal) / np.std(signal)
            noise_level = np.std(decomposition.resid) / np.std(signal)
            
            # Signal-to-noise ratio
            signal_to_noise = np.std(signal) / noise_level
            
            # Volatility
            volatility = np.std(np.diff(signal))
            
            # Momentum
            momentum = np.mean(np.diff(signal))
            
            # Mean reversion
            mean_reversion = -np.corrcoef(signal[:-1], signal[1:])[0,1]
            
            return SignalMetrics(
                stationarity=stationarity,
                entropy=entropy,
                kurtosis=kurt,
                skewness=skewness,
                normality=normality,
                autocorrelation=autocorr,
                trend_strength=trend_strength,
                seasonality_strength=seasonality_strength,
                noise_level=noise_level,
                signal_to_noise=signal_to_noise,
                volatility=volatility,
                momentum=momentum,
                mean_reversion=mean_reversion
            )
            
        except Exception as e:
            logger.error(f"Signal metrics computation error: {str(e)}")
            return SignalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def _detect_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect and handle outliers."""
        try:
            # Use z-score method
            z_scores = zscore(data)
            outliers = np.abs(z_scores) > 3
            
            # Count outliers
            outlier_count = np.sum(outliers)
            ALG_SIGNAL_OUTLIER_COUNT.inc(outlier_count)
            
            # Handle outliers (replace with median)
            if outlier_count > 0:
                median = np.median(data[~outliers])
                data[outliers] = median
                
            return data, outlier_count
            
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            return data, 0
            
    def _validate_inputs(self, theta: np.ndarray, phi: np.ndarray, acc: np.ndarray, vol: np.ndarray) -> bool:
        """Validate input arrays."""
        if not self.validation_enabled:
            return True
            
        try:
            # Check array shapes
            if not (theta.shape == phi.shape == acc.shape == vol.shape):
                logger.error("Input arrays must have the same shape")
                return False
                
            # Check for NaN values
            if np.isnan(theta).any() or np.isnan(phi).any() or np.isnan(acc).any() or np.isnan(vol).any():
                logger.warning("Input arrays contain NaN values")
                self.metadata['warning_count'] += 1
                
            # Check for infinite values
            if np.isinf(theta).any() or np.isinf(phi).any() or np.isinf(acc).any() or np.isinf(vol).any():
                logger.warning("Input arrays contain infinite values")
                self.metadata['warning_count'] += 1
                
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            self.metadata['error_count'] += 1
            return False
            
    def analyze_frequency_domain(self, signal: np.ndarray, sampling_rate: float = 1.0) -> FrequencyAnalysis:
        """Perform comprehensive frequency domain analysis."""
        try:
            # Compute FFT
            n = len(signal)
            fft_result = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(n, 1/sampling_rate)
            
            # Power spectrum
            power_spectrum = np.abs(fft_result)**2
            
            # Dominant frequency
            dominant_idx = np.argmax(power_spectrum)
            dominant_frequency = frequencies[dominant_idx]
            
            # Bandwidth
            total_power = np.sum(power_spectrum)
            cumulative_power = np.cumsum(power_spectrum)
            bandwidth = frequencies[np.where(cumulative_power >= 0.9 * total_power)[0][0]]
            
            # Spectral entropy
            normalized_spectrum = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
            
            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
            arithmetic_mean = np.mean(power_spectrum)
            spectral_flatness = geometric_mean / arithmetic_mean
            
            # Spectral centroid
            spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
            
            # Spectral rolloff
            rolloff_threshold = 0.85
            cumulative_power = np.cumsum(power_spectrum) / np.sum(power_spectrum)
            spectral_rolloff = frequencies[np.where(cumulative_power >= rolloff_threshold)[0][0]]
            
            return FrequencyAnalysis(
                frequencies=frequencies,
                power_spectrum=power_spectrum,
                dominant_frequency=dominant_frequency,
                bandwidth=bandwidth,
                spectral_entropy=spectral_entropy,
                spectral_flatness=spectral_flatness,
                spectral_centroid=spectral_centroid,
                spectral_rolloff=spectral_rolloff
            )
        except Exception as e:
            logger.error(f"Frequency domain analysis error: {str(e)}")
            return FrequencyAnalysis(
                np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            
    def compute_cross_correlation(self,
                                signal1: np.ndarray,
                                signal2: np.ndarray,
                                max_lag: Optional[int] = None) -> CrossCorrelation:
        """Compute cross-correlation between two signals."""
        try:
            if max_lag is None:
                max_lag = len(signal1) // 2
                
            # Compute cross-correlation
            correlation = np.correlate(signal1, signal2, mode='full')
            correlation = correlation[len(correlation)//2-max_lag:len(correlation)//2+max_lag+1]
            
            # Compute lags
            lag = np.arange(-max_lag, max_lag + 1)
            
            # Find maximum correlation and corresponding lag
            max_correlation = np.max(correlation)
            max_lag_idx = np.argmax(correlation)
            max_lag_value = lag[max_lag_idx]
            
            # Compute phase difference
            fft1 = np.fft.fft(signal1)
            fft2 = np.fft.fft(signal2)
            phase_difference = np.angle(fft1 * np.conj(fft2))
            phase_difference = np.mean(phase_difference)
            
            # Compute coherence
            cross_spectrum = fft1 * np.conj(fft2)
            power_spectrum1 = np.abs(fft1)**2
            power_spectrum2 = np.abs(fft2)**2
            coherence = np.abs(cross_spectrum)**2 / (power_spectrum1 * power_spectrum2)
            coherence = np.mean(coherence)
            
            return CrossCorrelation(
                correlation=correlation,
                lag=lag,
                max_correlation=max_correlation,
                max_lag=max_lag_value,
                phase_difference=phase_difference,
                coherence=coherence
            )
        except Exception as e:
            logger.error(f"Cross-correlation computation error: {str(e)}")
            return CrossCorrelation(
                np.array([]), np.array([]), 0.0, 0, 0.0, 0.0
            )
            
    def _detect_anomalies_isolation_forest(self, data: np.ndarray) -> AnomalyDetection:
        """Detect anomalies using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape data if needed
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # Fit Isolation Forest
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = -clf.fit_predict(data)
            
            # Set threshold at 95th percentile
            threshold = np.percentile(anomaly_scores, 95)
            
            # Identify anomalies
            anomalies = anomaly_scores > threshold
            anomaly_indices = np.where(anomalies)[0]
            
            # Compute confidence scores
            confidence_scores = anomaly_scores[anomalies]
            
            return AnomalyDetection(
                anomalies=anomalies,
                anomaly_scores=anomaly_scores,
                threshold=threshold,
                num_anomalies=np.sum(anomalies),
                anomaly_indices=anomaly_indices,
                confidence_scores=confidence_scores
            )
        except Exception as e:
            logger.error(f"Isolation Forest anomaly detection error: {str(e)}")
            return AnomalyDetection(
                np.array([]), np.array([]), 0.0, 0, np.array([]), np.array([])
            )
            
    def _detect_anomalies_lof(self, data: np.ndarray) -> AnomalyDetection:
        """Detect anomalies using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Reshape data if needed
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # Fit LOF
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            anomaly_scores = -clf.fit_predict(data)
            
            # Set threshold at 95th percentile
            threshold = np.percentile(anomaly_scores, 95)
            
            # Identify anomalies
            anomalies = anomaly_scores > threshold
            anomaly_indices = np.where(anomalies)[0]
            
            # Compute confidence scores
            confidence_scores = anomaly_scores[anomalies]
            
            return AnomalyDetection(
                anomalies=anomalies,
                anomaly_scores=anomaly_scores,
                threshold=threshold,
                num_anomalies=np.sum(anomalies),
                anomaly_indices=anomaly_indices,
                confidence_scores=confidence_scores
            )
        except Exception as e:
            logger.error(f"LOF anomaly detection error: {str(e)}")
            return AnomalyDetection(
                np.array([]), np.array([]), 0.0, 0, np.array([]), np.array([])
            )
            
    def _impute_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Impute missing values using advanced techniques."""
        try:
            from sklearn.impute import KNNImputer
            
            # Reshape data if needed
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # Use KNN imputer
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(data)
            
            return imputed_data.ravel() if len(data.shape) == 1 else imputed_data
        except Exception as e:
            logger.error(f"Missing value imputation error: {str(e)}")
            return data
            
    def _normalize_data(self, data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize data using various methods."""
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
                
            # Reshape data if needed
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                
            # Fit and transform
            normalized_data = scaler.fit_transform(data)
            
            return normalized_data.ravel() if len(data.shape) == 1 else normalized_data
        except Exception as e:
            logger.error(f"Data normalization error: {str(e)}")
            return data
            
    def assess_data_quality(self, data: np.ndarray) -> DataQualityMetrics:
        """Assess data quality using comprehensive metrics."""
        try:
            # Completeness
            completeness = 1.0 - np.mean(np.isnan(data))
            
            # Consistency
            std = np.std(data)
            mean = np.mean(data)
            consistency = 1.0 - (std / (mean + 1e-10))
            
            # Accuracy (using anomaly detection)
            anomaly_detection = self._detect_anomalies_isolation_forest(data)
            accuracy = 1.0 - (anomaly_detection.num_anomalies / len(data))
            
            # Timeliness (assuming data is time-ordered)
            time_diff = np.diff(data)
            timeliness = 1.0 - np.mean(np.abs(time_diff) > 3 * np.std(time_diff))
            
            # Validity (checking for valid ranges)
            valid_range = (np.percentile(data, 1), np.percentile(data, 99))
            validity = np.mean((data >= valid_range[0]) & (data <= valid_range[1]))
            
            # Uniqueness
            unique_values = len(np.unique(data))
            uniqueness = unique_values / len(data)
            
            # Integrity (checking for data corruption)
            checksum = np.sum(data)
            integrity = 1.0 if checksum != 0 else 0.0
            
            # Precision, Recall, and F1 Score (using anomaly detection)
            true_positives = anomaly_detection.num_anomalies
            false_positives = len(data) - true_positives
            false_negatives = 0  # Assuming no ground truth
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            return DataQualityMetrics(
                completeness=completeness,
                consistency=consistency,
                accuracy=accuracy,
                timeliness=timeliness,
                validity=validity,
                uniqueness=uniqueness,
                integrity=integrity,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
        except Exception as e:
            logger.error(f"Data quality assessment error: {str(e)}")
            return DataQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def _check_alert_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        try:
            for metric, value in metrics.items():
                if metric in self.alert_thresholds:
                    threshold = self.alert_thresholds[metric]
                    
                    if value > threshold:
                        # Determine alert level
                        if value > threshold * 2:
                            level = AlertLevel.CRITICAL
                        elif value > threshold * 1.5:
                            level = AlertLevel.ERROR
                        elif value > threshold * 1.2:
                            level = AlertLevel.WARNING
                        else:
                            level = AlertLevel.INFO
                            
                        # Create alert
                        alert = Alert(
                            timestamp=time.time(),
                            level=level.value,
                            message=f"{metric} exceeded threshold",
                            metric=metric,
                            value=value,
                            threshold=threshold,
                            action=self._get_alert_action(metric, level)
                        )
                        
                        alerts.append(alert)
                        
            return alerts
        except Exception as e:
            logger.error(f"Alert threshold check error: {str(e)}")
            return []
            
    def _get_alert_action(self, metric: str, level: AlertLevel) -> str:
        """Get recommended action for an alert."""
        actions = {
            'processing_time_ms': {
                AlertLevel.INFO: "Monitor processing time",
                AlertLevel.WARNING: "Consider optimizing code",
                AlertLevel.ERROR: "Review and optimize processing pipeline",
                AlertLevel.CRITICAL: "Emergency optimization required"
            },
            'memory_usage_gb': {
                AlertLevel.INFO: "Monitor memory usage",
                AlertLevel.WARNING: "Consider memory optimization",
                AlertLevel.ERROR: "Review memory-intensive operations",
                AlertLevel.CRITICAL: "Emergency memory cleanup required"
            },
            'gpu_memory_gb': {
                AlertLevel.INFO: "Monitor GPU memory usage",
                AlertLevel.WARNING: "Consider GPU memory optimization",
                AlertLevel.ERROR: "Review GPU operations",
                AlertLevel.CRITICAL: "Emergency GPU memory cleanup required"
            },
            'error_rate': {
                AlertLevel.INFO: "Monitor error rate",
                AlertLevel.WARNING: "Review recent errors",
                AlertLevel.ERROR: "Investigate error patterns",
                AlertLevel.CRITICAL: "Emergency error handling required"
            },
            'anomaly_rate': {
                AlertLevel.INFO: "Monitor anomaly rate",
                AlertLevel.WARNING: "Review anomaly patterns",
                AlertLevel.ERROR: "Investigate anomaly causes",
                AlertLevel.CRITICAL: "Emergency anomaly handling required"
            },
            'data_quality_score': {
                AlertLevel.INFO: "Monitor data quality",
                AlertLevel.WARNING: "Review data quality issues",
                AlertLevel.ERROR: "Investigate data quality problems",
                AlertLevel.CRITICAL: "Emergency data quality improvement required"
            },
            'signal_quality_score': {
                AlertLevel.INFO: "Monitor signal quality",
                AlertLevel.WARNING: "Review signal quality issues",
                AlertLevel.ERROR: "Investigate signal quality problems",
                AlertLevel.CRITICAL: "Emergency signal quality improvement required"
            }
        }
        
        return actions.get(metric, {}).get(level, "No specific action recommended")
        
    def _update_monitoring_metrics(self, metrics: Dict[str, float]) -> None:
        """Update monitoring metrics and check for alerts."""
        try:
            # Update Prometheus metrics
            if self.monitoring_enabled:
                for metric, value in metrics.items():
                    if metric == 'processing_time_ms':
                        ALG_SIGNAL_PROCESSING_DURATION.observe(value / 1000)
                    elif metric == 'memory_usage_gb':
                        ALG_SIGNAL_MEMORY_USAGE.set(value * 1024 * 1024 * 1024)
                    elif metric == 'gpu_memory_gb':
                        ALG_SIGNAL_GPU_MEMORY.set(value * 1024 * 1024 * 1024)
                    elif metric == 'signal_quality_score':
                        ALG_SIGNAL_QUALITY.set(value)
                    elif metric == 'data_quality_score':
                        ALG_SIGNAL_DATA_QUALITY.set(value)
                        
            # Check for alerts
            alerts = self._check_alert_thresholds(metrics)
            
            # Process alerts
            for alert in alerts:
                self.alert_history.append(alert)
                
                # Log alert
                if alert.level == AlertLevel.CRITICAL:
                    logger.critical(f"CRITICAL ALERT: {alert.message} (Value: {alert.value}, Threshold: {alert.threshold})")
                elif alert.level == AlertLevel.ERROR:
                    logger.error(f"ERROR ALERT: {alert.message} (Value: {alert.value}, Threshold: {alert.threshold})")
                elif alert.level == AlertLevel.WARNING:
                    logger.warning(f"WARNING ALERT: {alert.message} (Value: {alert.value}, Threshold: {alert.threshold})")
                else:
                    logger.info(f"INFO ALERT: {alert.message} (Value: {alert.value}, Threshold: {alert.threshold})")
                    
                # Take action if needed
                if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                    self._handle_critical_alert(alert)
                    
        except Exception as e:
            logger.error(f"Monitoring metrics update error: {str(e)}")
            
    def _handle_critical_alert(self, alert: Alert) -> None:
        """Handle critical alerts with appropriate actions."""
        try:
            if alert.metric == 'processing_time_ms':
                # Reduce batch size
                self.batch_size = max(100, self.batch_size // 2)
                logger.info(f"Reduced batch size to {self.batch_size}")
                
            elif alert.metric == 'memory_usage_gb':
                # Force garbage collection
                gc.collect()
                logger.info("Forced garbage collection")
                
            elif alert.metric == 'gpu_memory_gb':
                # Clear GPU cache
                if self.gpu_available:
                    cp.get_default_memory_pool().free_all_blocks()
                    logger.info("Cleared GPU memory cache")
                    
            elif alert.metric == 'error_rate':
                # Reset error count
                self.metadata['error_count'] = 0
                logger.info("Reset error count")
                
            elif alert.metric == 'anomaly_rate':
                # Adjust anomaly detection threshold
                self.alert_thresholds['anomaly_rate'] *= 1.5
                logger.info(f"Adjusted anomaly detection threshold to {self.alert_thresholds['anomaly_rate']}")
                
            elif alert.metric in ['data_quality_score', 'signal_quality_score']:
                # Switch to more robust processing method
                self.processing_method = SignalProcessingMethod.SAVITZKY_GOLAY
                logger.info("Switched to more robust processing method")
                
        except Exception as e:
            logger.error(f"Critical alert handling error: {str(e)}")
            
    def _load_audit_log(self) -> None:
        """Load audit log from file."""
        try:
            if os.path.exists(self.audit_log_path):
                with open(self.audit_log_path, 'r') as f:
                    self.audit_log = [AuditLog(**entry) for entry in json.load(f)]
        except Exception as e:
            logger.error(f"Audit log loading error: {str(e)}")
            self.audit_log = []
            
    def _save_audit_log(self) -> None:
        """Save audit log to file."""
        try:
            with open(self.audit_log_path, 'w') as f:
                json.dump([vars(entry) for entry in self.audit_log], f, indent=2)
        except Exception as e:
            logger.error(f"Audit log saving error: {str(e)}")
            
    def _log_audit_event(self, user: str, action: str, resource: str, status: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        try:
            log_entry = AuditLog(
                timestamp=time.time(),
                user=user,
                action=action,
                resource=resource,
                status=status,
                details=details
            )
            
            self.audit_log.append(log_entry)
            self._save_audit_log()
            
        except Exception as e:
            logger.error(f"Audit logging error: {str(e)}")
            
    def _encrypt_data(self, data: np.ndarray) -> bytes:
        """Encrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            # Generate key from encryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'quant_aero_log_salt',
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key))
            
            # Encrypt data
            f = Fernet(key)
            encrypted_data = f.encrypt(data.tobytes())
            
            return encrypted_data
        except Exception as e:
            logger.error(f"Data encryption error: {str(e)}")
            return data.tobytes()
            
    def _decrypt_data(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            
            # Generate key from encryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'quant_aero_log_salt',
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key))
            
            # Decrypt data
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)
            
            return np.frombuffer(decrypted_data, dtype=np.float64)
        except Exception as e:
            logger.error(f"Data decryption error: {str(e)}")
            return np.frombuffer(encrypted_data, dtype=np.float64)
            
    def _mask_sensitive_data(self, data: np.ndarray) -> np.ndarray:
        """Mask sensitive data for logging."""
        try:
            # Replace sensitive values with masked values
            masked_data = data.copy()
            sensitive_indices = np.where(np.abs(data) > np.percentile(np.abs(data), 95))[0]
            masked_data[sensitive_indices] = np.nan
            
            return masked_data
        except Exception as e:
            logger.error(f"Data masking error: {str(e)}")
            return data
            
    def compute_enhanced_alg_signal(self,
                                   theta: np.ndarray,
                                   phi: np.ndarray,
                                   acc: np.ndarray,
                                   vol: np.ndarray,
                                   params: Optional[SignalParameters] = None,
                                   user: str = "system") -> Tuple[np.ndarray, SignalMetrics]:
        """Compute enhanced algorithmic signal with comprehensive features."""
        start_time = time.time()
        ALG_SIGNAL_PROCESSING_COUNTER.inc()
        
        try:
            # Log access attempt
            self._log_audit_event(
                user=user,
                action="compute_signal",
                resource="algorithmic_signal",
                status="started",
                details={"timestamp": start_time}
            )
            
            # Validate inputs
            if not self._validate_inputs(theta, phi, acc, vol):
                logger.error("Input validation failed")
                ALG_SIGNAL_PROCESSING_ERRORS.inc()
                
                # Log validation failure
                self._log_audit_event(
                    user=user,
                    action="compute_signal",
                    resource="algorithmic_signal",
                    status="failed",
                    details={"error": "Input validation failed"}
                )
                
                return np.array([]), SignalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                
            # Encrypt sensitive data if security is enabled
            if self.security_enabled:
                theta_encrypted = self._encrypt_data(theta)
                phi_encrypted = self._encrypt_data(phi)
                acc_encrypted = self._encrypt_data(acc)
                vol_encrypted = self._encrypt_data(vol)
                
                # Store encrypted data
                self.metadata['encrypted_data'] = {
                    'theta': theta_encrypted,
                    'phi': phi_encrypted,
                    'acc': acc_encrypted,
                    'vol': vol_encrypted
                }
                
            # Assess data quality
            theta_quality = self.assess_data_quality(theta)
            phi_quality = self.assess_data_quality(phi)
            acc_quality = self.assess_data_quality(acc)
            vol_quality = self.assess_data_quality(vol)
            
            # Impute missing values if needed
            if theta_quality.completeness < 1.0:
                theta = self._impute_missing_values(theta)
            if phi_quality.completeness < 1.0:
                phi = self._impute_missing_values(phi)
            if acc_quality.completeness < 1.0:
                acc = self._impute_missing_values(acc)
            if vol_quality.completeness < 1.0:
                vol = self._impute_missing_values(vol)
                
            # Normalize data
            theta = self._normalize_data(theta)
            phi = self._normalize_data(phi)
            acc = self._normalize_data(acc)
            vol = self._normalize_data(vol)
            
            # Detect and handle anomalies
            theta_anomalies = self._detect_anomalies_isolation_forest(theta)
            phi_anomalies = self._detect_anomalies_isolation_forest(phi)
            acc_anomalies = self._detect_anomalies_isolation_forest(acc)
            vol_anomalies = self._detect_anomalies_isolation_forest(vol)
            
            # Handle anomalies
            if theta_anomalies.num_anomalies > 0:
                theta[theta_anomalies.anomaly_indices] = np.median(theta[~theta_anomalies.anomalies])
            if phi_anomalies.num_anomalies > 0:
                phi[phi_anomalies.anomaly_indices] = np.median(phi[~phi_anomalies.anomalies])
            if acc_anomalies.num_anomalies > 0:
                acc[acc_anomalies.anomaly_indices] = np.median(acc[~acc_anomalies.anomalies])
            if vol_anomalies.num_anomalies > 0:
                vol[vol_anomalies.anomaly_indices] = np.median(vol[~vol_anomalies.anomalies])
                
            # Use provided parameters or defaults
            p = params or self.params
            
            # Process signals
            if self.gpu_enabled:
                theta = self._process_signal_gpu(theta)
                phi = self._process_signal_gpu(phi)
                acc = self._process_signal_gpu(acc)
                vol = self._process_signal_gpu(vol)
                
            # Compute base signal components (using original function)
            base_signal = compute_alg_signal(theta, phi, acc, vol, p.a1, p.b1, p.g1)
            
            # Process signals based on configuration
            if self.parallel_enabled:
                processed_signals = self._process_parallel(np.column_stack((theta, phi, acc, vol)))
                theta, phi, acc, vol = processed_signals.T
            else:
                theta = self._process_batch(theta)
                phi = self._process_batch(phi)
                acc = self._process_batch(acc)
                vol = self._process_batch(vol)
            
            # Compute additional components
            A4 = p.alpha * np.tanh(theta * phi)  # Non-linear interaction
            A5 = p.beta * np.exp(-np.abs(acc))  # Exponential decay
            A6 = p.gamma * np.log1p(vol)  # Log transform
            A7 = p.delta * np.sin(2 * np.pi * theta)  # Periodic component
            A8 = p.epsilon * np.arctan(phi)  # Bounded component
            
            # Combine components
            signal = base_signal + A4 + A5 + A6 + A7 + A8
            
            # Apply rolling window smoothing
            if p.window_size > 1:
                signal = pd.Series(signal).rolling(window=p.window_size, min_periods=1).mean().values
                
            # Apply threshold
            signal = np.where(np.abs(signal) > p.threshold, signal, 0)
            
            # Perform frequency domain analysis
            freq_analysis = self.analyze_frequency_domain(signal)
            
            # Compute metrics
            metrics = self._compute_signal_metrics(signal)
            
            # Update metadata with frequency analysis
            self.metadata['frequency_analysis'] = {
                'dominant_frequency': freq_analysis.dominant_frequency,
                'bandwidth': freq_analysis.bandwidth,
                'spectral_entropy': freq_analysis.spectral_entropy,
                'spectral_flatness': freq_analysis.spectral_flatness,
                'spectral_centroid': freq_analysis.spectral_centroid,
                'spectral_rolloff': freq_analysis.spectral_rolloff
            }
            
            # Update metadata with data quality metrics
            self.metadata['data_quality'] = {
                'theta': theta_quality,
                'phi': phi_quality,
                'acc': acc_quality,
                'vol': vol_quality
            }
            
            # Update metadata with anomaly detection results
            self.metadata['anomalies'] = {
                'theta': theta_anomalies,
                'phi': phi_anomalies,
                'acc': acc_anomalies,
                'vol': vol_anomalies
            }
            
            # Update monitoring metrics
            metrics = {
                'processing_time_ms': (time.time() - start_time) * 1000,
                'memory_usage_gb': psutil.Process().memory_info().rss / (1024 * 1024 * 1024),
                'gpu_memory_gb': cp.cuda.Device(0).mem_info[0] / (1024 * 1024 * 1024) if self.gpu_available else 0,
                'error_rate': self.metadata['error_count'] / (self.metadata['error_count'] + 1),
                'anomaly_rate': sum(a.num_anomalies for a in [theta_anomalies, phi_anomalies, acc_anomalies, vol_anomalies]) / (4 * len(theta)),
                'data_quality_score': min(q.f1_score for q in [theta_quality, phi_quality, acc_quality, vol_quality]),
                'signal_quality_score': metrics.signal_to_noise
            }
            
            self._update_monitoring_metrics(metrics)
            
            # Log successful computation
            self._log_audit_event(
                user=user,
                action="compute_signal",
                resource="algorithmic_signal",
                status="completed",
                details={
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "signal_metrics": vars(metrics),
                    "data_quality": self.metadata.get('data_quality', {}),
                    "anomalies": self.metadata.get('anomalies', {}),
                    "frequency_analysis": self.metadata.get('frequency_analysis', {})
                }
            )
            
            return signal, metrics
            
        except Exception as e:
            logger.error(f"Algorithmic signal computation error: {str(e)}")
            ALG_SIGNAL_PROCESSING_ERRORS.inc()
            self.metadata['error_count'] += 1
            
            # Log error
            self._log_audit_event(
                user=user,
                action="compute_signal",
                resource="algorithmic_signal",
                status="error",
                details={"error": str(e)}
            )
            
            return np.array([]), SignalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def build_enhanced_alg_signal_from_df(self,
                                         df: pd.DataFrame,
                                         params: Optional[SignalParameters] = None) -> Tuple[pd.Series, SignalMetrics]:
        """
        Build enhanced algorithmic signal from DataFrame with comprehensive features.
        """
        try:
            # Extract required columns
            theta = df['theta'].values
            phi = df['phi'].values
            acc = df['acc'].values
            vol = df['vol'].values
            
            # Compute signal
            signal, metrics = self.compute_enhanced_alg_signal(theta, phi, acc, vol, params)
            
            # Convert to Series
            signal_series = pd.Series(signal, index=df.index)
            
            return signal_series, metrics
            
        except Exception as e:
            logger.error(f"DataFrame signal construction error: {str(e)}")
            return pd.Series(), SignalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def run_performance_benchmark(self, n_samples: int = 1000000) -> Dict[str, float]:
        """Run comprehensive performance benchmark."""
        try:
            # Generate test data
            theta = np.random.normal(0, 1, n_samples)
            phi = np.random.normal(0, 1, n_samples)
            acc = np.random.normal(0, 1, n_samples)
            vol = np.random.normal(0, 1, n_samples)
            
            # Benchmark metrics
            metrics = {}
            
            # CPU processing benchmark
            start_time = time.time()
            self._process_signal_cpu(theta)
            metrics['cpu_processing_time_ms'] = (time.time() - start_time) * 1000
            
            # GPU processing benchmark
            if self.gpu_available:
                start_time = time.time()
                self._process_signal_gpu(theta)
                metrics['gpu_processing_time_ms'] = (time.time() - start_time) * 1000
                
            # Batch processing benchmark
            start_time = time.time()
            self._process_batch(theta)
            metrics['batch_processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Parallel processing benchmark
            if self.ray_available:
                start_time = time.time()
                self._process_parallel(theta)
                metrics['parallel_processing_time_ms'] = (time.time() - start_time) * 1000
                
            # Memory usage benchmark
            process = psutil.Process()
            start_memory = process.memory_info().rss
            self.compute_enhanced_alg_signal(theta, phi, acc, vol)
            end_memory = process.memory_info().rss
            metrics['memory_usage_mb'] = (end_memory - start_memory) / (1024 * 1024)
            
            # GPU memory benchmark
            if self.gpu_available:
                start_memory = cp.cuda.Device(0).mem_info[0]
                self.compute_enhanced_alg_signal(theta, phi, acc, vol)
                end_memory = cp.cuda.Device(0).mem_info[0]
                metrics['gpu_memory_usage_mb'] = (end_memory - start_memory) / (1024 * 1024)
                
            return metrics
        except Exception as e:
            logger.error(f"Performance benchmark error: {str(e)}")
            return {}
            
    def run_unit_tests(self) -> Dict[str, bool]:
        """Run comprehensive unit tests."""
        try:
            test_results = {}
            
            # Test data generation
            n = 1000
            theta = np.linspace(0, 2*np.pi, n)
            phi = np.sin(theta) + np.random.normal(0, 0.1, n)
            acc = np.cos(theta) + np.random.normal(0, 0.1, n)
            vol = np.abs(np.sin(2*theta)) + np.random.normal(0, 0.1, n)
            
            # Test signal computation
            signal, metrics = self.compute_enhanced_alg_signal(theta, phi, acc, vol)
            test_results['signal_computation'] = len(signal) == n
            
            # Test data quality assessment
            quality = self.assess_data_quality(theta)
            test_results['data_quality_assessment'] = all(v > 0 for v in vars(quality).values())
            
            # Test anomaly detection
            anomalies = self._detect_anomalies_isolation_forest(theta)
            test_results['anomaly_detection'] = len(anomalies.anomalies) > 0
            
            # Test frequency analysis
            freq_analysis = self.analyze_frequency_domain(signal)
            test_results['frequency_analysis'] = len(freq_analysis.frequencies) > 0
            
            # Test cross-correlation
            correlation = self.compute_cross_correlation(theta, phi)
            test_results['cross_correlation'] = len(correlation.correlation) > 0
            
            # Test data normalization
            normalized = self._normalize_data(theta)
            test_results['data_normalization'] = np.allclose(np.mean(normalized), 0, atol=1e-10)
            
            # Test missing value imputation
            theta_with_nan = theta.copy()
            theta_with_nan[::10] = np.nan
            imputed = self._impute_missing_values(theta_with_nan)
            test_results['missing_value_imputation'] = not np.isnan(imputed).any()
            
            # Test security features
            if self.security_enabled:
                encrypted = self._encrypt_data(theta)
                decrypted = self._decrypt_data(encrypted)
                test_results['encryption_decryption'] = np.allclose(theta, decrypted)
                
                masked = self._mask_sensitive_data(theta)
                test_results['data_masking'] = np.isnan(masked).any()
                
            # Test monitoring
            if self.monitoring_enabled:
                self._update_monitoring_metrics({
                    'processing_time_ms': 100,
                    'memory_usage_gb': 1,
                    'error_rate': 0.01
                })
                test_results['monitoring'] = len(self.alert_history) > 0
                
            # Test audit logging
            if self.security_enabled:
                self._log_audit_event("test_user", "test_action", "test_resource", "test_status", {})
                test_results['audit_logging'] = len(self.audit_log) > 0
                
            return test_results
        except Exception as e:
            logger.error(f"Unit test error: {str(e)}")
            return {}
            
    def run_validation_tests(self, ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Run validation tests against ground truth."""
        try:
            validation_results = {}
            
            # Generate test data
            n = 1000
            theta = np.linspace(0, 2*np.pi, n)
            phi = np.sin(theta) + np.random.normal(0, 0.1, n)
            acc = np.cos(theta) + np.random.normal(0, 0.1, n)
            vol = np.abs(np.sin(2*theta)) + np.random.normal(0, 0.1, n)
            
            # Compute signal
            signal, metrics = self.compute_enhanced_alg_signal(theta, phi, acc, vol)
            
            # If ground truth is provided, compare with it
            if ground_truth is not None:
                # Mean squared error
                mse = np.mean((signal - ground_truth)**2)
                validation_results['mean_squared_error'] = mse
                
                # Root mean squared error
                rmse = np.sqrt(mse)
                validation_results['root_mean_squared_error'] = rmse
                
                # Mean absolute error
                mae = np.mean(np.abs(signal - ground_truth))
                validation_results['mean_absolute_error'] = mae
                
                # R-squared
                ss_res = np.sum((signal - ground_truth)**2)
                ss_tot = np.sum((ground_truth - np.mean(ground_truth))**2)
                r2 = 1 - (ss_res / ss_tot)
                validation_results['r_squared'] = r2
                
            # Validate signal properties
            validation_results['stationarity'] = metrics.stationarity
            validation_results['entropy'] = metrics.entropy
            validation_results['signal_to_noise'] = metrics.signal_to_noise
            
            # Validate data quality
            quality = self.assess_data_quality(signal)
            validation_results['data_quality'] = quality.f1_score
            
            # Validate anomaly detection
            anomalies = self._detect_anomalies_isolation_forest(signal)
            validation_results['anomaly_rate'] = anomalies.num_anomalies / len(signal)
            
            return validation_results
        except Exception as e:
            logger.error(f"Validation test error: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = AlgorithmicSignalGenerator(
        signal_type=SignalType.ANGULAR,
        processing_method=SignalProcessingMethod.WAVELET,
        validation_enabled=True,
        monitoring_enabled=True,
        cache_enabled=True,
        gpu_enabled=True,
        security_enabled=True
    )
    
    # Run performance benchmark
    benchmark_results = generator.run_performance_benchmark()
    print("Performance Benchmark Results:")
    for metric, value in benchmark_results.items():
        print(f"{metric}: {value:.2f}")
        
    # Run unit tests
    test_results = generator.run_unit_tests()
    print("\nUnit Test Results:")
    for test, passed in test_results.items():
        print(f"{test}: {'PASSED' if passed else 'FAILED'}")
        
    # Run validation tests
    validation_results = generator.run_validation_tests()
    print("\nValidation Test Results:")
    for metric, value in validation_results.items():
        print(f"{metric}: {value:.4f}")

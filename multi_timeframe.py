# ==========================
# multi_timeframe.py
# ==========================
# Enhanced multi-resolution angular signal generator with comprehensive features.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import time
from tqdm import tqdm
import warnings
from scipy import signal
from scipy.stats import zscore, kurtosis, skew, normaltest
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pywt
import numba
from prometheus_client import Counter, Gauge, Histogram, Summary
import psutil
import gc
import dask.dataframe as dd
import cupy as cp
import ray
from ray.util.multiprocessing import Pool
import json
import os
from pathlib import Path
import hashlib
import uuid
import tempfile
import shutil
from functools import partial
import inspect
import sys

# Local imports
from angles_metrics import compute_angles
from log_signal import compute_log_signal
from alg_signal import compute_alg_signal
from config_env import (
    DEBUG_MODE,
    MTF_RESAMPLING_METHOD,
    MTF_INTERPOLATION_METHOD,
    MTF_VALIDATION_ENABLED,
    MTF_MONITORING_ENABLED,
    MTF_PERFORMANCE_MONITORING_ENABLED,
    MTF_CACHE_ENABLED,
    MTF_PARALLEL_PROCESSING_ENABLED,
    MTF_GPU_ENABLED,
    MTF_STREAMING_ENABLED,
    MTF_DATA_QUALITY_ENABLED,
    MTF_SECURITY_ENABLED
)

logger = logging.getLogger(__name__)

# Prometheus metrics
MTF_PROCESSING_COUNTER = Counter('mtf_processing_total', 'Total number of MTF signals processed')
MTF_PROCESSING_ERRORS = Counter('mtf_processing_errors_total', 'Total number of MTF processing errors')
MTF_PROCESSING_DURATION = Histogram('mtf_processing_duration_seconds', 'Time spent processing MTF signals')
MTF_SIGNAL_QUALITY = Gauge('mtf_signal_quality', 'Quality score of MTF signals')
MTF_MEMORY_USAGE = Gauge('mtf_memory_usage_bytes', 'Memory usage during MTF processing')
MTF_PARALLEL_TASKS = Gauge('mtf_parallel_tasks', 'Number of parallel tasks')
MTF_GPU_MEMORY = Gauge('mtf_gpu_memory_bytes', 'GPU memory usage')
MTF_DATA_QUALITY = Gauge('mtf_data_quality_score', 'Data quality score')
MTF_SIGNAL_STATIONARITY = Gauge('mtf_signal_stationarity', 'Signal stationarity score')
MTF_SIGNAL_ENTROPY = Gauge('mtf_signal_entropy', 'Signal entropy')
MTF_OUTLIER_COUNT = Counter('mtf_outliers_total', 'Total number of outliers detected')

class ResamplingMethod(Enum):
    """Supported resampling methods."""
    LAST = 'last'
    MEAN = 'mean'
    MEDIAN = 'median'
    OHLC = 'ohlc'
    VWAP = 'vwap'  # Volume-weighted average price
    TICK = 'tick'  # Tick-based resampling
    CUSTOM = 'custom'

class InterpolationMethod(Enum):
    """Supported interpolation methods."""
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'
    SPLINE = 'spline'
    NEAREST = 'nearest'
    ZERO = 'zero'
    PCHIP = 'pchip'
    AKIMA = 'akima'
    TIME = 'time'  # Time-based interpolation

class SignalProcessingMethod(Enum):
    """Supported signal processing methods."""
    WAVELET = 'wavelet'
    FFT = 'fft'
    SAVITZKY_GOLAY = 'savitzky_golay'
    KALMAN = 'kalman'
    HODRICK_PRESCOTT = 'hodrick_prescott'
    CUSTOM = 'custom'

class DataQualityMetric(Enum):
    """Supported data quality metrics."""
    COMPLETENESS = 'completeness'
    CONSISTENCY = 'consistency'
    ACCURACY = 'accuracy'
    TIMELINESS = 'timeliness'
    VALIDITY = 'validity'
    UNIQUENESS = 'uniqueness'

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

@dataclass
class MTFMetadata:
    """Enhanced metadata for multi-timeframe signal processing."""
    timestamp: datetime
    source: str
    resolutions: List[int]
    processing_time_ms: float
    error_count: int
    warning_count: int
    signal_quality: float
    custom_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    signal_metrics: Dict[str, SignalMetrics]
    data_quality_metrics: Dict[str, float]
    security_metrics: Dict[str, Any]

class MultiTimeframeGenerator:
    """Enhanced multi-timeframe signal generator with comprehensive features."""
    
    def __init__(self,
                 resampling_method: ResamplingMethod = ResamplingMethod.LAST,
                 interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
                 signal_processing_method: SignalProcessingMethod = SignalProcessingMethod.WAVELET,
                 validation_enabled: bool = MTF_VALIDATION_ENABLED,
                 monitoring_enabled: bool = MTF_MONITORING_ENABLED,
                 cache_enabled: bool = MTF_CACHE_ENABLED,
                 parallel_processing_enabled: bool = MTF_PARALLEL_PROCESSING_ENABLED,
                 gpu_enabled: bool = MTF_GPU_ENABLED,
                 streaming_enabled: bool = MTF_STREAMING_ENABLED,
                 data_quality_enabled: bool = MTF_DATA_QUALITY_ENABLED,
                 security_enabled: bool = MTF_SECURITY_ENABLED):
        """
        Initialize the multi-timeframe generator with enhanced features.
        """
        self.resampling_method = resampling_method
        self.interpolation_method = interpolation_method
        self.signal_processing_method = signal_processing_method
        self.validation_enabled = validation_enabled
        self.monitoring_enabled = monitoring_enabled
        self.cache_enabled = cache_enabled
        self.parallel_processing_enabled = parallel_processing_enabled
        self.gpu_enabled = gpu_enabled
        self.streaming_enabled = streaming_enabled
        self.data_quality_enabled = data_quality_enabled
        self.security_enabled = security_enabled
        
        # Initialize components
        self._setup_parallel_processing()
        self._setup_gpu()
        self._setup_cache()
        self._setup_metadata()
        self._setup_security()
        
    def _setup_parallel_processing(self) -> None:
        """Setup parallel processing components."""
        if self.parallel_processing_enabled:
            ray.init(ignore_reinit_error=True)
            self.pool = Pool()
            
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
        self.metadata = MTFMetadata(
            timestamp=datetime.now(),
            source='',
            resolutions=[],
            processing_time_ms=0,
            error_count=0,
            warning_count=0,
            signal_quality=1.0,
            custom_metadata={},
            performance_metrics={},
            validation_metrics={},
            signal_metrics={},
            data_quality_metrics={},
            security_metrics={}
        )
        
    def _setup_security(self) -> None:
        """Setup security components."""
        if self.security_enabled:
            self.encryption_key = self._generate_encryption_key()
            self.access_control = self._setup_access_control()
            
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
        
    def _process_signal_gpu(self, data: np.ndarray) -> np.ndarray:
        """Process signal using GPU acceleration."""
        if not self.gpu_available:
            return data
            
        try:
            # Convert to GPU array
            gpu_data = cp.asarray(data)
            
            # Process on GPU
            if self.signal_processing_method == SignalProcessingMethod.WAVELET:
                # Perform wavelet transform on GPU
                coeffs = pywt.wavedec(gpu_data, 'db1', level=3)
                processed = pywt.waverec(coeffs, 'db1')
            elif self.signal_processing_method == SignalProcessingMethod.FFT:
                # Perform FFT on GPU
                fft_data = cp.fft.fft(gpu_data)
                processed = cp.fft.ifft(fft_data).real
            else:
                processed = gpu_data
                
            # Convert back to CPU
            return cp.asnumpy(processed)
            
        except Exception as e:
            logger.error(f"GPU processing error: {str(e)}")
            return data
            
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
            
            return SignalMetrics(
                stationarity=stationarity,
                entropy=entropy,
                kurtosis=kurt,
                skewness=skewness,
                normality=normality,
                autocorrelation=autocorr,
                trend_strength=trend_strength,
                seasonality_strength=seasonality_strength,
                noise_level=noise_level
            )
            
        except Exception as e:
            logger.error(f"Signal metrics computation error: {str(e)}")
            return SignalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def _detect_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect and handle outliers."""
        try:
            # Use z-score method
            z_scores = zscore(data)
            outliers = np.abs(z_scores) > 3
            
            # Count outliers
            outlier_count = np.sum(outliers)
            MTF_OUTLIER_COUNT.inc(outlier_count)
            
            # Handle outliers (replace with median)
            if outlier_count > 0:
                median = np.median(data[~outliers])
                data[outliers] = median
                
            return data, outlier_count
            
        except Exception as e:
            logger.error(f"Outlier detection error: {str(e)}")
            return data, 0
            
    def _process_resolution_parallel(self, args: Tuple[pd.DataFrame, str, int]) -> Dict[str, pd.Series]:
        """Process a single resolution in parallel."""
        df, column, resolution = args
        return self._process_single_resolution(df, column, resolution)
        
    def _process_single_resolution(self, df: pd.DataFrame, column: str, resolution: int) -> Dict[str, pd.Series]:
        """Process data for a single resolution."""
        try:
            # Resample data
            sub_df = self._resample_data(df, column, resolution)
            
            # Detect and handle outliers
            data = sub_df[column].values
            data, outlier_count = self._detect_outliers(data)
            sub_df[column] = data
            
            # Process signal
            if self.gpu_enabled:
                processed_data = self._process_signal_gpu(data)
            else:
                processed_data = data
                
            # Compute angles
            theta, phi, acc, vol = compute_angles(sub_df, column)
            
            # Compute signals
            log_sig = compute_log_signal(theta, phi, acc, vol)
            alg_sig = compute_alg_signal(theta, phi, acc, vol)
            
            # Compute metrics
            signal_metrics = self._compute_signal_metrics(processed_data)
            
            return {
                f'log_{resolution}m': pd.Series(log_sig, index=sub_df.index),
                f'alg_{resolution}m': pd.Series(alg_sig, index=sub_df.index),
                f'metrics_{resolution}m': signal_metrics
            }
            
        except Exception as e:
            logger.error(f"Resolution processing error: {str(e)}")
            return {}
            
    async def compute_signals(self,
                            df: pd.DataFrame,
                            column: str = 'price',
                            resolutions: List[int] = [1,5,10,60,180,300,600,1440,2880,10080,43200],
                            metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Compute multi-timeframe signals with enhanced features.
        """
        start_time = time.time()
        self.metadata.timestamp = datetime.now()
        self.metadata.source = column
        self.metadata.resolutions = resolutions
        self.metadata.custom_metadata = metadata or {}
        
        try:
            # Update Prometheus metrics
            MTF_PROCESSING_COUNTER.inc()
            
            # Validate data
            if not self._validate_data(df, column):
                logger.error("Data validation failed")
                MTF_PROCESSING_ERRORS.inc()
                return pd.DataFrame()
                
            # Initialize results
            results = {}
            signal_metrics = {}
            
            # Process resolutions
            if self.parallel_processing_enabled:
                # Prepare arguments for parallel processing
                args = [(df, column, r) for r in resolutions]
                
                # Process in parallel
                with self.pool as pool:
                    resolution_results = pool.map(self._process_resolution_parallel, args)
                    
                # Combine results
                for res in resolution_results:
                    results.update({k: v for k, v in res.items() if not k.startswith('metrics_')})
                    signal_metrics.update({k: v for k, v in res.items() if k.startswith('metrics_')})
                    
            else:
                # Process sequentially
                for r in tqdm(resolutions, desc="Processing resolutions"):
                    res = self._process_single_resolution(df, column, r)
                    results.update({k: v for k, v in res.items() if not k.startswith('metrics_')})
                    signal_metrics.update({k: v for k, v in res.items() if k.startswith('metrics_')})
                    
            # Update metadata
            self.metadata.signal_metrics = signal_metrics
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.metadata.processing_time_ms = processing_time
            MTF_PROCESSING_DURATION.observe(processing_time / 1000)
            MTF_MEMORY_USAGE.set(psutil.Process().memory_info().rss)
            
            if self.monitoring_enabled:
                self.metadata.performance_metrics = self._monitor_performance()
                
            if DEBUG_MODE:
                logger.info(f"MTF signals computed successfully")
                logger.info(f"Processing time: {processing_time:.2f}ms")
                logger.info(f"Signal metrics: {signal_metrics}")
                logger.info(f"Performance metrics: {self.metadata.performance_metrics}")
                
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"MTF signal computation error: {str(e)}")
            MTF_PROCESSING_ERRORS.inc()
            self.metadata.error_count += 1
            return pd.DataFrame()
            
    def clear_cache(self) -> None:
        """Clear the signal cache."""
        if self.cache_enabled:
            self._cache.clear()
            self._cache_metadata.clear()
            gc.collect()

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=10000, freq='1min')
    data = pd.DataFrame({
        'price': np.random.randn(10000).cumsum(),
        'volume': np.random.randint(100, 1000, 10000)
    }, index=dates)
    
    # Compute signals with enhanced features
    generator = MultiTimeframeGenerator(
        resampling_method=ResamplingMethod.VWAP,
        interpolation_method=InterpolationMethod.CUBIC,
        signal_processing_method=SignalProcessingMethod.WAVELET,
        validation_enabled=True,
        monitoring_enabled=True,
        cache_enabled=True,
        parallel_processing_enabled=True,
        gpu_enabled=True,
        streaming_enabled=False,
        data_quality_enabled=True,
        security_enabled=True
    )
    
    signals = asyncio.run(generator.compute_signals(
        data,
        column='price',
        resolutions=[1,5,15,30,60,240,1440],
        metadata={
            'source': 'test',
            'version': '1.0',
            'environment': 'production',
            'tags': ['test', 'mtf']
        }
    ))

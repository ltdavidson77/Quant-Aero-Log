# ==========================
# label_generator.py
# ==========================
# Creates multi-horizon labels based on future price returns for classification tasks.

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
import json

# Import system components
from physics import (
    QuantumState,
    QuantumStateManager,
    QuantumStateType,
    QuantumStateMetadata,
    QuantumDecoherence,
    QuantumEntropy,
    QuantumResonance,
    QuantumChaos,
    QuantumFractal,
    QuantumNeural,
    QuantumEvolution,
    QuantumOptimization,
    QuantumLearning,
    TwistedHamiltonian
)
from storage.db_manager import get_db_session, LabelStorage, QuantumStateStorage
from storage.snapshot_rotator import rotate_snapshots
from monitoring.metrics import (
    record_label_metrics,
    LabelQualityMetrics,
    SystemMetrics,
    QuantumMetrics
)
from monitoring.alerting import AlertManager, QuantumAlertManager
from utils.validation import validate_input_data, validate_quantum_state
from utils.performance import PerformanceMonitor, QuantumPerformanceMonitor
from utils.cache import CacheManager, CacheConfig, QuantumCacheManager
from utils.logger import get_logger
from config.label_config import LabelConfig, LabelType, QuantumLabelConfig
from config.system_config import SystemConfig
from api_clients.api_router import fetch_market_data, fetch_quantum_data
from algorithms.predictive_analysis import PredictiveAnalyzer
from algorithms.quantum_trigonometric import QuantumTrigonometricAnalyzer
from algorithms.quantum_evolution import QuantumEvolutionAnalyzer
from algorithms.quantum_learning import QuantumLearningAnalyzer

logger = get_logger("label_generator")

class LabelGenerationMode(Enum):
    """Different modes of label generation."""
    STANDARD = "standard"
    QUANTUM_ENHANCED = "quantum_enhanced"
    ADAPTIVE = "adaptive"
    PROBABILISTIC = "probabilistic"
    PHYSICS_BASED = "physics_based"
    HYBRID = "hybrid"
    QUANTUM_OPTIMIZED = "quantum_optimized"

@dataclass
class LabelMetrics:
    """Comprehensive metrics for label generation results."""
    quality: float
    class_balance: Dict[str, float]
    prediction_horizon: int
    generation_time: float
    quantum_enhancement_factor: float
    confidence_scores: Dict[str, float]
    label_distribution: Dict[str, float]
    volatility_adjustment: float
    trend_strength: float
    noise_level: float
    physics_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    storage_metrics: Dict[str, float]
    api_metrics: Dict[str, float]
    quantum_metrics: Dict[str, float]

class LabelGenerator:
    """Enhanced label generator with full system integration and quantum optimization."""
    
    def __init__(self,
                 config: LabelConfig,
                 system_config: SystemConfig,
                 quantum_config: Optional[Dict[str, Any]] = None):
        """
        Initialize label generator with comprehensive configuration.
        
        Args:
            config: Label configuration parameters
            system_config: System configuration parameters
            quantum_config: Optional quantum optimization configuration
        """
        self.config = config
        self.system_config = system_config
        self.quantum_config = quantum_config or QuantumLabelConfig()
        
        # Initialize quantum components
        self.quantum_state_manager = QuantumStateManager(quantum_config)
        self.quantum_cache = QuantumCacheManager()
        self.quantum_storage = QuantumStateStorage(get_db_session())
        self.quantum_performance = QuantumPerformanceMonitor()
        self.quantum_alert_manager = QuantumAlertManager()
        
        # Initialize system components
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager(CacheConfig())
        self.alert_manager = AlertManager()
        self.db_session = get_db_session()
        self.label_storage = LabelStorage(self.db_session)
        
        # Initialize analyzers
        self.predictive_analyzer = PredictiveAnalyzer()
        self.quantum_trigonometric = QuantumTrigonometricAnalyzer()
        self.quantum_evolution = QuantumEvolutionAnalyzer()
        self.quantum_learning = QuantumLearningAnalyzer()
        
        # Initialize history tracking
        self.label_history = []
        self.metrics_history = []
        self.quantum_state_history = []
        self.system_metrics = SystemMetrics()
        
    async def generate_labels(self,
                            df: pd.DataFrame,
                            price_col: str = 'Close',
                            horizon: Optional[int] = None) -> Tuple[pd.Series, LabelMetrics]:
        """
        Generate enhanced labels with comprehensive metrics and system integration.
        
        Args:
            df: DataFrame containing price data
            price_col: Column name for price data
            horizon: Optional prediction horizon in minutes
            
        Returns:
            Tuple of (label Series, label metrics)
        """
        with self.performance_monitor.measure():
            try:
                # Validate inputs
                validate_input_data(df, price_col)
                
                # Use configured horizon if none provided
                horizon = horizon or self.config.default_horizon
                
                # Fetch additional market data if needed
                market_data = await self._fetch_market_data(df.index)
                quantum_data = await self._fetch_quantum_data(df.index)
                
                # Calculate future returns
                future_prices = df[price_col].shift(-horizon)
                returns = (future_prices - df[price_col]) / df[price_col]
                
                # Apply quantum optimization
                quantum_state = await self._prepare_quantum_state(returns, quantum_data)
                optimized_returns = await self._apply_quantum_optimization(quantum_state, returns)
                
                # Apply physics-based analysis
                physics_metrics = await self._apply_physics_analysis(optimized_returns, market_data)
                
                # Generate labels based on mode
                if self.config.generation_mode == LabelGenerationMode.QUANTUM_OPTIMIZED:
                    labels = await self._generate_quantum_optimized_labels(optimized_returns, physics_metrics)
                elif self.config.generation_mode == LabelGenerationMode.PHYSICS_BASED:
                    labels = await self._generate_physics_based_labels(optimized_returns, physics_metrics)
                elif self.config.generation_mode == LabelGenerationMode.HYBRID:
                    labels = await self._generate_hybrid_labels(optimized_returns, physics_metrics)
                else:
                    labels = self._generate_standard_labels(optimized_returns)
                
                # Calculate metrics
                metrics = await self._calculate_comprehensive_metrics(labels, optimized_returns, horizon, physics_metrics, quantum_state)
                
                # Store labels, metrics, and quantum states
                await self._store_labels_and_metrics(labels, metrics, quantum_state)
                
                # Update system metrics
                self._update_system_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                return pd.Series(labels, index=df.index), metrics
                
            except Exception as e:
                logger.error(f"Error generating labels: {str(e)}")
                self.alert_manager.trigger_alert("label_generation_error", str(e))
                self.quantum_alert_manager.trigger_alert("quantum_generation_error", str(e))
                raise
                
    async def _prepare_quantum_state(self,
                                   returns: pd.Series,
                                   quantum_data: Dict[str, Any]) -> QuantumState:
        """Prepare quantum state for optimization."""
        try:
            # Check cache for existing state
            cache_key = self._generate_quantum_cache_key(returns, quantum_data)
            cached_state = self.quantum_cache.get(cache_key)
            
            if cached_state is not None:
                return cached_state
                
            # Create new quantum state
            quantum_state = QuantumState(returns.values)
            
            # Apply quantum data
            quantum_state.apply_quantum_data(quantum_data)
            
            # Store in cache
            self.quantum_cache.set(cache_key, quantum_state)
            
            return quantum_state
            
        except Exception as e:
            logger.error(f"Error preparing quantum state: {str(e)}")
            raise
            
    async def _apply_quantum_optimization(self,
                                        quantum_state: QuantumState,
                                        returns: pd.Series) -> pd.Series:
        """Apply quantum optimization to returns."""
        try:
            # Apply quantum gates
            quantum_state.apply_quantum_gates()
            
            # Apply quantum learning
            quantum_state = self.quantum_learning.optimize(quantum_state)
            
            # Measure and return optimized returns
            optimized_values = quantum_state.measure()
            return pd.Series(optimized_values, index=returns.index)
            
        except Exception as e:
            logger.error(f"Error applying quantum optimization: {str(e)}")
            return returns
            
    async def _generate_quantum_optimized_labels(self,
                                               returns: pd.Series,
                                               physics_metrics: Dict[str, Any]) -> np.ndarray:
        """Generate labels using quantum-optimized approach."""
        try:
            # Apply quantum trigonometric analysis
            trig_analysis = self.quantum_trigonometric.analyze(returns.values)
            
            # Apply quantum evolution analysis
            evolution_analysis = self.quantum_evolution.analyze(returns.values)
            
            # Apply quantum learning analysis
            learning_analysis = self.quantum_learning.analyze(returns.values)
            
            # Combine analyses with quantum weights
            combined_signal = (
                trig_analysis['signal'] * self.quantum_config.trig_weight +
                evolution_analysis['signal'] * self.quantum_config.evolution_weight +
                learning_analysis['signal'] * self.quantum_config.learning_weight
            )
            
            # Generate labels with quantum thresholds
            labels = np.zeros(len(returns))
            labels[combined_signal > self.quantum_config.up_threshold] = 1
            labels[combined_signal < self.quantum_config.down_threshold] = -1
            
            return labels
            
        except Exception as e:
            logger.error(f"Error generating quantum-optimized labels: {str(e)}")
            return np.zeros(len(returns))
            
    async def _store_labels_and_metrics(self,
                                      labels: np.ndarray,
                                      metrics: LabelMetrics,
                                      quantum_state: QuantumState) -> None:
        """Store labels, metrics, and quantum states in the database."""
        try:
            # Store labels
            await self.label_storage.store_labels(
                labels=labels,
                horizon=metrics.prediction_horizon,
                generation_time=datetime.utcnow(),
                metrics=metrics.__dict__
            )
            
            # Store quantum state
            await self.quantum_storage.store_state(
                state=quantum_state,
                metrics=metrics.quantum_metrics
            )
            
            # Rotate snapshots if needed
            await rotate_snapshots(self.db_session)
            
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            self.alert_manager.trigger_alert("storage_error", str(e))
            self.quantum_alert_manager.trigger_alert("quantum_storage_error", str(e))
            
    def _update_system_metrics(self, metrics: LabelMetrics) -> None:
        """Update system-wide metrics."""
        try:
            self.system_metrics.update({
                'label_generation_time': metrics.generation_time,
                'label_quality': metrics.quality,
                'quantum_enhancement': metrics.quantum_enhancement_factor,
                'system_performance': metrics.system_metrics['performance'],
                'storage_usage': metrics.storage_metrics['usage'],
                'api_latency': metrics.api_metrics['latency'],
                'quantum_performance': metrics.quantum_metrics['performance']
            })
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
            
    def _check_alerts(self, metrics: LabelMetrics) -> None:
        """Check for system and quantum alerts."""
        try:
            # Check quality alerts
            if metrics.quality < self.system_config.quality_threshold:
                self.alert_manager.trigger_alert(
                    "low_quality_labels",
                    f"Label quality below threshold: {metrics.quality}"
                )
                
            # Check quantum alerts
            if metrics.quantum_metrics['performance'] < self.quantum_config.performance_threshold:
                self.quantum_alert_manager.trigger_alert(
                    "low_quantum_performance",
                    f"Quantum performance below threshold: {metrics.quantum_metrics['performance']}"
                )
                
            # Check system performance alerts
            if metrics.system_metrics['performance'] < self.system_config.performance_threshold:
                self.alert_manager.trigger_alert(
                    "low_performance",
                    f"System performance below threshold: {metrics.system_metrics['performance']}"
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")

# ----------------------------------------
# Multi-Horizon Label Set Builder
# ----------------------------------------
async def generate_multi_horizon_labels(
    df: pd.DataFrame,
    price_col: str = 'Close',
    horizons: Optional[List[int]] = None,
    config: Optional[LabelConfig] = None,
    system_config: Optional[SystemConfig] = None,
    quantum_config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[int, LabelMetrics]]:
    """
    Generate multi-horizon labels with comprehensive configuration and system integration.
    
    Args:
        df: DataFrame containing price data
        price_col: Column name for price data
        horizons: Optional list of prediction horizons
        config: Optional label configuration
        system_config: Optional system configuration
        quantum_config: Optional quantum configuration
        
    Returns:
        Tuple of (label DataFrame, metrics dictionary)
    """
    config = config or LabelConfig()
    system_config = system_config or SystemConfig()
    quantum_config = quantum_config or QuantumLabelConfig()
    
    generator = LabelGenerator(config, system_config, quantum_config)
    
    # Use configured horizons if none provided
    horizons = horizons or config.default_horizons
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=df.index)
    metrics_dict = {}
    
    # Generate labels for each horizon in parallel
    tasks = []
    for horizon in horizons:
        tasks.append(generator.generate_labels(
            df,
            price_col=price_col,
            horizon=horizon
        ))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Process results
    for horizon, (labels, metrics) in zip(horizons, results):
        result[f'label_{horizon}m'] = labels
        metrics_dict[horizon] = metrics
        
    return result, metrics_dict

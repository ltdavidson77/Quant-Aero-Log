# ==========================
# risk/market_regime.py
# ==========================
# Market regime detection and analysis.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from statsmodels.tsa.statespace.tools import is_invertible
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager, RiskMetricType
from risk.portfolio_optimizer import PortfolioOptimizer, OptimizationType
from risk.risk_allocator import RiskAllocator, AllocationType

logger = get_logger("market_regime")

class RegimeType(Enum):
    """Types of market regimes."""
    NORMAL = "normal"            # Normal market conditions
    VOLATILE = "volatile"        # High volatility regime
    TRENDING = "trending"        # Strong trend regime
    MEAN_REVERSION = "mean_reversion"  # Mean reversion regime
    CRASH = "crash"             # Market crash regime
    RECOVERY = "recovery"       # Market recovery regime
    SIDEWAYS = "sideways"       # Sideways market regime
    MOMENTUM = "momentum"       # Momentum regime
    RANGE_BOUND = "range_bound" # Range-bound regime
    BREAKOUT = "breakout"       # Breakout regime
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Liquidity crisis regime
    RISK_ON = "risk_on"         # Risk-on regime
    RISK_OFF = "risk_off"       # Risk-off regime
    DEFLATIONARY = "deflationary"  # Deflationary regime
    INFLATIONARY = "inflationary"  # Inflationary regime
    RECESSION = "recession"     # Recession regime
    EXPANSION = "expansion"     # Economic expansion regime

class DetectionMethod(Enum):
    """Methods for regime detection."""
    GMM = "gmm"                # Gaussian Mixture Model
    KMEANS = "kmeans"          # K-means clustering
    THRESHOLD = "threshold"    # Threshold-based detection
    HMM = "hmm"               # Hidden Markov Model
    MARKOV = "markov"         # Markov switching model
    WAVELET = "wavelet"       # Wavelet analysis
    FRACTAL = "fractal"       # Fractal analysis
    MACHINE_LEARNING = "ml"   # Machine learning approach
    ENSEMBLE = "ensemble"     # Ensemble of methods

@dataclass
class RegimeConfig:
    """Configuration for market regime detection."""
    detection_method: DetectionMethod = DetectionMethod.GMM
    lookback_window: int = 252  # Lookback window
    min_observations: int = 20  # Minimum observations
    confidence_level: float = 0.95  # Confidence level
    min_regime_duration: int = 5  # Minimum regime duration
    n_regimes: int = 3  # Number of regimes
    feature_weights: Dict[str, float] = None  # Feature weights
    regime_thresholds: Dict[str, float] = None  # Regime thresholds
    volatility_threshold: float = 0.2  # Volatility threshold
    trend_threshold: float = 0.1  # Trend threshold
    momentum_threshold: float = 0.05  # Momentum threshold
    mean_reversion_threshold: float = 0.02  # Mean reversion threshold
    range_threshold: float = 0.15  # Range threshold
    breakout_threshold: float = 0.1  # Breakout threshold
    liquidity_threshold: float = 0.3  # Liquidity threshold
    risk_threshold: float = 0.2  # Risk threshold
    inflation_threshold: float = 0.05  # Inflation threshold
    recession_threshold: float = -0.02  # Recession threshold

class MarketRegimeManager:
    """Manages market regime detection and analysis."""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        if self.config.feature_weights is None:
            self.config.feature_weights = {
                'returns': 0.2,
                'volatility': 0.2,
                'trend': 0.15,
                'momentum': 0.15,
                'mean_reversion': 0.1,
                'range': 0.1,
                'breakout': 0.1
            }
        if self.config.regime_thresholds is None:
            self.config.regime_thresholds = {
                'volatile': 0.2,
                'trending': 0.1,
                'mean_reversion': 0.02,
                'crash': -0.1,
                'recovery': 0.1,
                'sideways': 0.05,
                'momentum': 0.05,
                'range_bound': 0.15,
                'breakout': 0.1,
                'liquidity_crisis': 0.3,
                'risk_on': 0.2,
                'risk_off': -0.2,
                'deflationary': -0.02,
                'inflationary': 0.05,
                'recession': -0.02,
                'expansion': 0.02
            }
        self._setup_market_regime_manager()
    
    def _setup_market_regime_manager(self) -> None:
        """Initialize market regime manager."""
        try:
            logger.info("Initializing market regime manager")
            self._validate_config()
            self._setup_features()
        except Exception as e:
            logger.error(f"Failed to setup market regime manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate regime configuration."""
        try:
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.min_regime_duration <= 0:
                raise ValueError("Minimum regime duration must be positive")
            if self.config.n_regimes < 2:
                raise ValueError("Number of regimes must be at least 2")
            if sum(self.config.feature_weights.values()) != 1:
                raise ValueError("Feature weights must sum to 1")
            for threshold in self.config.regime_thresholds.values():
                if threshold <= 0:
                    raise ValueError("Regime thresholds must be positive")
        except Exception as e:
            logger.error(f"Invalid regime configuration: {str(e)}")
            raise
    
    def _setup_features(self) -> None:
        """Setup feature calculation methods."""
        try:
            self.feature_methods = {
                'returns': self._calculate_returns,
                'volatility': self._calculate_volatility,
                'trend': self._calculate_trend,
                'momentum': self._calculate_momentum,
                'mean_reversion': self._calculate_mean_reversion,
                'range': self._calculate_range,
                'breakout': self._calculate_breakout,
                'liquidity': self._calculate_liquidity,
                'risk': self._calculate_risk,
                'inflation': self._calculate_inflation,
                'recession': self._calculate_recession
            }
        except Exception as e:
            logger.error(f"Failed to setup features: {str(e)}")
            raise
    
    def detect_regime(self,
                     returns: pd.DataFrame,
                     risk_metrics_manager: Optional[RiskMetricsManager] = None,
                     portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                     risk_allocator: Optional[RiskAllocator] = None) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics_manager: Risk metrics manager (optional)
            portfolio_optimizer: Portfolio optimizer (optional)
            risk_allocator: Risk allocator (optional)
            
        Returns:
            Dictionary containing regime detection results
        """
        try:
            # Calculate features
            features = self._calculate_features(returns)
            
            # Detect regime
            if self.config.detection_method == DetectionMethod.GMM:
                regime = self._detect_regime_gmm(features)
            elif self.config.detection_method == DetectionMethod.KMEANS:
                regime = self._detect_regime_kmeans(features)
            elif self.config.detection_method == DetectionMethod.THRESHOLD:
                regime = self._detect_regime_threshold(features)
            elif self.config.detection_method == DetectionMethod.HMM:
                regime = self._detect_regime_hmm(features)
            elif self.config.detection_method == DetectionMethod.MARKOV:
                regime = self._detect_regime_markov(features)
            elif self.config.detection_method == DetectionMethod.WAVELET:
                regime = self._detect_regime_wavelet(features)
            elif self.config.detection_method == DetectionMethod.FRACTAL:
                regime = self._detect_regime_fractal(features)
            elif self.config.detection_method == DetectionMethod.MACHINE_LEARNING:
                regime = self._detect_regime_ml(features)
            elif self.config.detection_method == DetectionMethod.ENSEMBLE:
                regime = self._detect_regime_ensemble(features)
            else:
                raise ValueError(f"Unsupported detection method: {self.config.detection_method}")
            
            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(regime, features)
            
            # Calculate regime transitions
            transitions = self._calculate_regime_transitions(regime)
            
            # Calculate regime probabilities
            probabilities = self._calculate_regime_probabilities(regime, features)
            
            # Calculate regime duration
            duration = self._calculate_regime_duration(regime)
            
            # Calculate regime impact
            impact = self._calculate_regime_impact(regime, returns)
            
            return {
                'regime': regime,
                'metrics': metrics,
                'transitions': transitions,
                'probabilities': probabilities,
                'duration': duration,
                'impact': impact
            }
            
        except Exception as e:
            logger.error(f"Failed to detect regime: {str(e)}")
            raise
    
    def _calculate_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection."""
        try:
            features = pd.DataFrame(index=returns.index)
            
            # Calculate basic features
            for feature, weight in self.config.feature_weights.items():
                if feature in self.feature_methods:
                    features[feature] = self.feature_methods[feature](returns) * weight
            
            # Calculate advanced features
            features['liquidity'] = self._calculate_liquidity(returns)
            features['risk'] = self._calculate_risk(returns)
            features['inflation'] = self._calculate_inflation(returns)
            features['recession'] = self._calculate_recession(returns)
            
            # Normalize features
            scaler = StandardScaler()
            features = pd.DataFrame(
                scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to calculate features: {str(e)}")
            raise
    
    def _calculate_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate returns feature."""
        try:
            return returns.mean(axis=1)
        except Exception as e:
            logger.error(f"Failed to calculate returns: {str(e)}")
            raise
    
    def _calculate_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate volatility feature."""
        try:
            return returns.std(axis=1) * np.sqrt(252)
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {str(e)}")
            raise
    
    def _calculate_trend(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate trend feature."""
        try:
            trend = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    trend.iloc[i] = np.nan
                else:
                    trend.iloc[i] = np.polyfit(
                        range(self.config.lookback_window),
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1),
                        1
                    )[0]
            return trend
        except Exception as e:
            logger.error(f"Failed to calculate trend: {str(e)}")
            raise
    
    def _calculate_momentum(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate momentum feature."""
        try:
            momentum = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    momentum.iloc[i] = np.nan
                else:
                    momentum.iloc[i] = returns.iloc[i-self.config.lookback_window:i].mean(axis=1).mean()
            return momentum
        except Exception as e:
            logger.error(f"Failed to calculate momentum: {str(e)}")
            raise
    
    def _calculate_mean_reversion(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion feature."""
        try:
            mean_reversion = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    mean_reversion.iloc[i] = np.nan
                else:
                    mean_reversion.iloc[i] = -np.corrcoef(
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1),
                        range(self.config.lookback_window)
                    )[0, 1]
            return mean_reversion
        except Exception as e:
            logger.error(f"Failed to calculate mean reversion: {str(e)}")
            raise
    
    def _calculate_range(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate range feature."""
        try:
            range_ = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    range_.iloc[i] = np.nan
                else:
                    range_.iloc[i] = (
                        returns.iloc[i-self.config.lookback_window:i].max(axis=1).max() -
                        returns.iloc[i-self.config.lookback_window:i].min(axis=1).min()
                    )
            return range_
        except Exception as e:
            logger.error(f"Failed to calculate range: {str(e)}")
            raise
    
    def _calculate_breakout(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate breakout feature."""
        try:
            breakout = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    breakout.iloc[i] = np.nan
                else:
                    breakout.iloc[i] = (
                        returns.iloc[i].mean() -
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1).mean()
                    ) / returns.iloc[i-self.config.lookback_window:i].std(axis=1).mean()
            return breakout
        except Exception as e:
            logger.error(f"Failed to calculate breakout: {str(e)}")
            raise
    
    def _calculate_liquidity(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate liquidity feature."""
        try:
            liquidity = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    liquidity.iloc[i] = np.nan
                else:
                    liquidity.iloc[i] = (
                        returns.iloc[i-self.config.lookback_window:i].std(axis=1).mean() /
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1).abs().mean()
                    )
            return liquidity
        except Exception as e:
            logger.error(f"Failed to calculate liquidity: {str(e)}")
            raise
    
    def _calculate_risk(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate risk feature."""
        try:
            risk = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    risk.iloc[i] = np.nan
                else:
                    risk.iloc[i] = (
                        returns.iloc[i-self.config.lookback_window:i].std(axis=1).mean() *
                        np.sqrt(252)
                    )
            return risk
        except Exception as e:
            logger.error(f"Failed to calculate risk: {str(e)}")
            raise
    
    def _calculate_inflation(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate inflation feature."""
        try:
            inflation = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    inflation.iloc[i] = np.nan
                else:
                    inflation.iloc[i] = (
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1).mean() *
                        252
                    )
            return inflation
        except Exception as e:
            logger.error(f"Failed to calculate inflation: {str(e)}")
            raise
    
    def _calculate_recession(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate recession feature."""
        try:
            recession = pd.Series(index=returns.index)
            for i in range(len(returns)):
                if i < self.config.lookback_window:
                    recession.iloc[i] = np.nan
                else:
                    recession.iloc[i] = (
                        returns.iloc[i-self.config.lookback_window:i].mean(axis=1).mean() *
                        252
                    )
            return recession
        except Exception as e:
            logger.error(f"Failed to calculate recession: {str(e)}")
            raise
    
    def _detect_regime_gmm(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using Gaussian Mixture Model."""
        try:
            gmm = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type='full',
                random_state=42
            )
            gmm.fit(features)
            regime = pd.Series(
                gmm.predict(features),
                index=features.index
            )
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using GMM: {str(e)}")
            raise
    
    def _detect_regime_kmeans(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using K-means clustering."""
        try:
            kmeans = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42
            )
            kmeans.fit(features)
            regime = pd.Series(
                kmeans.predict(features),
                index=features.index
            )
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using K-means: {str(e)}")
            raise
    
    def _detect_regime_threshold(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using threshold-based method."""
        try:
            regime = pd.Series(index=features.index)
            
            for i in range(len(features)):
                if features['volatility'].iloc[i] > self.config.volatility_threshold:
                    regime.iloc[i] = RegimeType.VOLATILE.value
                elif features['trend'].iloc[i] > self.config.trend_threshold:
                    regime.iloc[i] = RegimeType.TRENDING.value
                elif features['mean_reversion'].iloc[i] > self.config.mean_reversion_threshold:
                    regime.iloc[i] = RegimeType.MEAN_REVERSION.value
                elif features['range'].iloc[i] < self.config.range_threshold:
                    regime.iloc[i] = RegimeType.RANGE_BOUND.value
                elif features['breakout'].iloc[i] > self.config.breakout_threshold:
                    regime.iloc[i] = RegimeType.BREAKOUT.value
                elif features['liquidity'].iloc[i] > self.config.liquidity_threshold:
                    regime.iloc[i] = RegimeType.LIQUIDITY_CRISIS.value
                elif features['risk'].iloc[i] > self.config.risk_threshold:
                    regime.iloc[i] = RegimeType.RISK_ON.value
                elif features['risk'].iloc[i] < -self.config.risk_threshold:
                    regime.iloc[i] = RegimeType.RISK_OFF.value
                elif features['inflation'].iloc[i] > self.config.inflation_threshold:
                    regime.iloc[i] = RegimeType.INFLATIONARY.value
                elif features['inflation'].iloc[i] < -self.config.inflation_threshold:
                    regime.iloc[i] = RegimeType.DEFLATIONARY.value
                elif features['recession'].iloc[i] < self.config.recession_threshold:
                    regime.iloc[i] = RegimeType.RECESSION.value
                elif features['recession'].iloc[i] > -self.config.recession_threshold:
                    regime.iloc[i] = RegimeType.EXPANSION.value
                else:
                    regime.iloc[i] = RegimeType.NORMAL.value
            
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using threshold method: {str(e)}")
            raise
    
    def _detect_regime_hmm(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using Hidden Markov Model."""
        try:
            # Implement HMM detection
            # This is a placeholder for actual HMM implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using HMM: {str(e)}")
            raise
    
    def _detect_regime_markov(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using Markov switching model."""
        try:
            # Implement Markov switching model detection
            # This is a placeholder for actual Markov switching implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using Markov switching: {str(e)}")
            raise
    
    def _detect_regime_wavelet(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using wavelet analysis."""
        try:
            # Implement wavelet analysis detection
            # This is a placeholder for actual wavelet implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using wavelet analysis: {str(e)}")
            raise
    
    def _detect_regime_fractal(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using fractal analysis."""
        try:
            # Implement fractal analysis detection
            # This is a placeholder for actual fractal implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using fractal analysis: {str(e)}")
            raise
    
    def _detect_regime_ml(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using machine learning approach."""
        try:
            # Implement machine learning detection
            # This is a placeholder for actual ML implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using machine learning: {str(e)}")
            raise
    
    def _detect_regime_ensemble(self, features: pd.DataFrame) -> pd.Series:
        """Detect regime using ensemble of methods."""
        try:
            # Implement ensemble detection
            # This is a placeholder for actual ensemble implementation
            regime = pd.Series(index=features.index)
            return regime
        except Exception as e:
            logger.error(f"Failed to detect regime using ensemble: {str(e)}")
            raise
    
    def _calculate_regime_metrics(self, regime: pd.Series, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime metrics."""
        try:
            metrics = {}
            
            for regime_type in RegimeType:
                regime_mask = regime == regime_type.value
                if regime_mask.any():
                    metrics[regime_type.value] = {
                        'count': regime_mask.sum(),
                        'duration': regime_mask.sum() / len(regime),
                        'returns': features['returns'][regime_mask].mean(),
                        'volatility': features['volatility'][regime_mask].mean(),
                        'trend': features['trend'][regime_mask].mean(),
                        'momentum': features['momentum'][regime_mask].mean(),
                        'mean_reversion': features['mean_reversion'][regime_mask].mean(),
                        'range': features['range'][regime_mask].mean(),
                        'breakout': features['breakout'][regime_mask].mean(),
                        'liquidity': features['liquidity'][regime_mask].mean(),
                        'risk': features['risk'][regime_mask].mean(),
                        'inflation': features['inflation'][regime_mask].mean(),
                        'recession': features['recession'][regime_mask].mean()
                    }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to calculate regime metrics: {str(e)}")
            raise
    
    def _calculate_regime_transitions(self, regime: pd.Series) -> pd.DataFrame:
        """Calculate regime transitions."""
        try:
            transitions = pd.DataFrame(
                0,
                index=[r.value for r in RegimeType],
                columns=[r.value for r in RegimeType]
            )
            
            for i in range(1, len(regime)):
                from_regime = regime.iloc[i-1]
                to_regime = regime.iloc[i]
                transitions.loc[from_regime, to_regime] += 1
            
            return transitions
        except Exception as e:
            logger.error(f"Failed to calculate regime transitions: {str(e)}")
            raise
    
    def _calculate_regime_probabilities(self, regime: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime probabilities."""
        try:
            probabilities = pd.DataFrame(index=features.index)
            
            for regime_type in RegimeType:
                regime_mask = regime == regime_type.value
                if regime_mask.any():
                    probabilities[regime_type.value] = regime_mask.astype(float)
            
            return probabilities
        except Exception as e:
            logger.error(f"Failed to calculate regime probabilities: {str(e)}")
            raise
    
    def _calculate_regime_duration(self, regime: pd.Series) -> Dict[str, Any]:
        """Calculate regime duration."""
        try:
            duration = {}
            
            for regime_type in RegimeType:
                regime_mask = regime == regime_type.value
                if regime_mask.any():
                    duration[regime_type.value] = {
                        'mean': regime_mask.sum() / len(regime),
                        'min': regime_mask.astype(int).groupby(
                            (regime_mask != regime_mask.shift()).cumsum()
                        ).sum().min(),
                        'max': regime_mask.astype(int).groupby(
                            (regime_mask != regime_mask.shift()).cumsum()
                        ).sum().max()
                    }
            
            return duration
        except Exception as e:
            logger.error(f"Failed to calculate regime duration: {str(e)}")
            raise
    
    def _calculate_regime_impact(self, regime: pd.Series, returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime impact."""
        try:
            impact = {}
            
            for regime_type in RegimeType:
                regime_mask = regime == regime_type.value
                if regime_mask.any():
                    impact[regime_type.value] = {
                        'returns': returns[regime_mask].mean().mean(),
                        'volatility': returns[regime_mask].std().mean() * np.sqrt(252),
                        'sharpe_ratio': (
                            returns[regime_mask].mean().mean() /
                            returns[regime_mask].std().mean() * np.sqrt(252)
                        ),
                        'max_drawdown': returns[regime_mask].cumsum().min().min(),
                        'win_rate': (returns[regime_mask] > 0).mean().mean(),
                        'profit_factor': abs(
                            returns[regime_mask][returns[regime_mask] > 0].sum().sum() /
                            returns[regime_mask][returns[regime_mask] < 0].sum().sum()
                        )
                    }
            
            return impact
        except Exception as e:
            logger.error(f"Failed to calculate regime impact: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        n_assets = 5
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (252, n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Initialize market regime manager
        config = RegimeConfig(
            detection_method=DetectionMethod.GMM,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            min_regime_duration=5,
            n_regimes=3,
            feature_weights={
                'returns': 0.2,
                'volatility': 0.2,
                'trend': 0.15,
                'momentum': 0.15,
                'mean_reversion': 0.1,
                'range': 0.1,
                'breakout': 0.1
            },
            regime_thresholds={
                'volatile': 0.2,
                'trending': 0.1,
                'mean_reversion': 0.02,
                'crash': -0.1,
                'recovery': 0.1,
                'sideways': 0.05,
                'momentum': 0.05,
                'range_bound': 0.15,
                'breakout': 0.1,
                'liquidity_crisis': 0.3,
                'risk_on': 0.2,
                'risk_off': -0.2,
                'deflationary': -0.02,
                'inflationary': 0.05,
                'recession': -0.02,
                'expansion': 0.02
            },
            volatility_threshold=0.2,
            trend_threshold=0.1,
            momentum_threshold=0.05,
            mean_reversion_threshold=0.02,
            range_threshold=0.15,
            breakout_threshold=0.1,
            liquidity_threshold=0.3,
            risk_threshold=0.2,
            inflation_threshold=0.05,
            recession_threshold=-0.02
        )
        
        manager = MarketRegimeManager(config)
        
        # Detect regime
        result = manager.detect_regime(returns)
        
        # Print results
        print("\nMarket Regime Analysis Results:")
        print("\nCurrent Regime:")
        print(result['regime'].iloc[-1])
        
        print("\nRegime Metrics:")
        print(result['metrics'])
        
        print("\nRegime Transitions:")
        print(result['transitions'])
        
        print("\nRegime Probabilities:")
        print(result['probabilities'].iloc[-1])
        
        print("\nRegime Duration:")
        print(result['duration'])
        
        print("\nRegime Impact:")
        print(result['impact'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 
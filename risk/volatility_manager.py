# ==========================
# risk/volatility_manager.py
# ==========================
# Volatility analysis and management.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from arch import arch_model
from statsmodels.tsa.statespace.tools import cfa_simulation_smoother
from utils.logger import get_logger
from risk.market_regime import MarketRegimeManager, RegimeType

logger = get_logger("volatility_manager")

class VolatilityType(Enum):
    """Types of volatility analysis."""
    SIMPLE = "simple"        # Simple historical volatility
    EWMA = "ewma"           # Exponentially weighted moving average
    GARCH = "garch"         # GARCH model
    EGARCH = "egarch"       # EGARCH model
    GJR_GARCH = "gjr_garch" # GJR-GARCH model
    STOCHASTIC = "stochastic" # Stochastic volatility
    REALIZED = "realized"   # Realized volatility
    IMPLIED = "implied"     # Implied volatility
    FORECAST = "forecast"   # Volatility forecasting
    REGIME = "regime"       # Regime-based volatility

@dataclass
class VolatilityConfig:
    """Configuration for volatility analysis."""
    volatility_type: VolatilityType = VolatilityType.SIMPLE
    lookback_window: int = 252  # Lookback window for volatility calculation
    min_observations: int = 20  # Minimum observations required
    confidence_level: float = 0.95  # Confidence level for volatility bands
    ewma_lambda: float = 0.94  # Lambda parameter for EWMA
    garch_order: Tuple[int, int] = (1, 1)  # (p, q) order for GARCH models
    realized_frequency: str = "1D"  # Frequency for realized volatility
    forecast_horizon: int = 10  # Forecast horizon in days
    regime_aware: bool = True  # Whether to consider market regimes
    regime_threshold: float = 0.7  # Threshold for regime classification
    volatility_bands: List[float] = None  # Custom volatility bands
    significance_threshold: float = 0.05  # Significance threshold for tests

class VolatilityManager:
    """Manages volatility analysis and management."""
    
    def __init__(self, config: VolatilityConfig):
        self.config = config
        if self.config.volatility_bands is None:
            self.config.volatility_bands = [0.5, 1.0, 1.5, 2.0]
        self._setup_volatility_manager()
    
    def _setup_volatility_manager(self) -> None:
        """Initialize volatility manager."""
        try:
            logger.info("Initializing volatility manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup volatility manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate volatility configuration."""
        try:
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.ewma_lambda <= 0 or self.config.ewma_lambda >= 1:
                raise ValueError("EWMA lambda must be between 0 and 1")
            if self.config.garch_order[0] < 1 or self.config.garch_order[1] < 1:
                raise ValueError("GARCH order must be at least (1,1)")
            if self.config.forecast_horizon < 1:
                raise ValueError("Forecast horizon must be at least 1")
            if self.config.regime_threshold <= 0 or self.config.regime_threshold >= 1:
                raise ValueError("Regime threshold must be between 0 and 1")
            if self.config.significance_threshold <= 0 or self.config.significance_threshold >= 1:
                raise ValueError("Significance threshold must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid volatility configuration: {str(e)}")
            raise
    
    def analyze_volatility(self,
                         returns: pd.DataFrame,
                         regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """
        Analyze volatility of assets.
        
        Args:
            returns: DataFrame of asset returns
            regime_manager: Market regime manager (optional)
            
        Returns:
            Dictionary containing volatility analysis results
        """
        try:
            if self.config.volatility_type == VolatilityType.SIMPLE:
                return self._analyze_simple_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.EWMA:
                return self._analyze_ewma_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.GARCH:
                return self._analyze_garch_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.EGARCH:
                return self._analyze_egarch_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.GJR_GARCH:
                return self._analyze_gjr_garch_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.STOCHASTIC:
                return self._analyze_stochastic_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.REALIZED:
                return self._analyze_realized_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.IMPLIED:
                return self._analyze_implied_volatility(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.FORECAST:
                return self._analyze_volatility_forecast(returns, regime_manager)
            elif self.config.volatility_type == VolatilityType.REGIME:
                return self._analyze_regime_volatility(returns, regime_manager)
            else:
                raise ValueError(f"Unsupported volatility type: {self.config.volatility_type}")
        except Exception as e:
            logger.error(f"Failed to analyze volatility: {str(e)}")
            raise
    
    def _analyze_simple_volatility(self,
                                 returns: pd.DataFrame,
                                 regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze simple historical volatility."""
        try:
            # Calculate volatility
            volatility = returns.rolling(
                self.config.lookback_window
            ).std() * np.sqrt(252)
            
            # Calculate confidence intervals
            n = self.config.lookback_window
            z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
            ci_lower = volatility * (1 - z_score / np.sqrt(2 * n))
            ci_upper = volatility * (1 + z_score / np.sqrt(2 * n))
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'confidence_intervals': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze simple volatility: {str(e)}")
            raise
    
    def _analyze_ewma_volatility(self,
                                returns: pd.DataFrame,
                                regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze EWMA volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            # Calculate EWMA volatility
            for asset in returns.columns:
                squared_returns = returns[asset] ** 2
                ewma = squared_returns.ewm(alpha=1-self.config.ewma_lambda).mean()
                volatility[asset] = np.sqrt(ewma * 252)
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze EWMA volatility: {str(e)}")
            raise
    
    def _analyze_garch_volatility(self,
                                 returns: pd.DataFrame,
                                 regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze GARCH volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            parameters = {}
            
            # Fit GARCH model for each asset
            for asset in returns.columns:
                model = arch_model(
                    returns[asset],
                    p=self.config.garch_order[0],
                    q=self.config.garch_order[1]
                )
                result = model.fit(disp='off')
                volatility[asset] = result.conditional_volatility * np.sqrt(252)
                parameters[asset] = {
                    'omega': result.params['omega'],
                    'alpha': result.params['alpha[1]'],
                    'beta': result.params['beta[1]']
                }
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'parameters': parameters,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze GARCH volatility: {str(e)}")
            raise
    
    def _analyze_egarch_volatility(self,
                                  returns: pd.DataFrame,
                                  regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze EGARCH volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            parameters = {}
            
            # Fit EGARCH model for each asset
            for asset in returns.columns:
                model = arch_model(
                    returns[asset],
                    p=self.config.garch_order[0],
                    q=self.config.garch_order[1],
                    o=1
                )
                result = model.fit(disp='off')
                volatility[asset] = result.conditional_volatility * np.sqrt(252)
                parameters[asset] = {
                    'omega': result.params['omega'],
                    'alpha': result.params['alpha[1]'],
                    'gamma': result.params['gamma[1]'],
                    'beta': result.params['beta[1]']
                }
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'parameters': parameters,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze EGARCH volatility: {str(e)}")
            raise
    
    def _analyze_gjr_garch_volatility(self,
                                     returns: pd.DataFrame,
                                     regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze GJR-GARCH volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            parameters = {}
            
            # Fit GJR-GARCH model for each asset
            for asset in returns.columns:
                model = arch_model(
                    returns[asset],
                    p=self.config.garch_order[0],
                    q=self.config.garch_order[1],
                    o=1,
                    power=1.0
                )
                result = model.fit(disp='off')
                volatility[asset] = result.conditional_volatility * np.sqrt(252)
                parameters[asset] = {
                    'omega': result.params['omega'],
                    'alpha': result.params['alpha[1]'],
                    'gamma': result.params['gamma[1]'],
                    'beta': result.params['beta[1]']
                }
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'parameters': parameters,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze GJR-GARCH volatility: {str(e)}")
            raise
    
    def _analyze_stochastic_volatility(self,
                                     returns: pd.DataFrame,
                                     regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze stochastic volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            # Fit stochastic volatility model for each asset
            for asset in returns.columns:
                # Use simulation smoother
                smoother = cfa_simulation_smoother(returns[asset])
                volatility[asset] = np.sqrt(smoother.smoothed_state_cov * 252)
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze stochastic volatility: {str(e)}")
            raise
    
    def _analyze_realized_volatility(self,
                                   returns: pd.DataFrame,
                                   regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze realized volatility."""
        try:
            # Resample returns to higher frequency
            high_freq_returns = returns.resample(self.config.realized_frequency).sum()
            
            # Calculate realized volatility
            volatility = high_freq_returns.rolling(
                self.config.lookback_window
            ).std() * np.sqrt(252)
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze realized volatility: {str(e)}")
            raise
    
    def _analyze_implied_volatility(self,
                                  returns: pd.DataFrame,
                                  regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze implied volatility."""
        try:
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            # Calculate implied volatility (placeholder)
            # In practice, this would use option prices
            for asset in returns.columns:
                # Use historical volatility as proxy
                volatility[asset] = returns[asset].rolling(
                    self.config.lookback_window
                ).std() * np.sqrt(252)
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'volatility': volatility,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze implied volatility: {str(e)}")
            raise
    
    def _analyze_volatility_forecast(self,
                                   returns: pd.DataFrame,
                                   regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze volatility forecasts."""
        try:
            # Initialize results
            forecast = pd.DataFrame(index=range(self.config.forecast_horizon), columns=returns.columns)
            
            # Forecast volatility for each asset
            for asset in returns.columns:
                # Fit GARCH model
                model = arch_model(
                    returns[asset],
                    p=self.config.garch_order[0],
                    q=self.config.garch_order[1]
                )
                result = model.fit(disp='off')
                
                # Generate forecasts
                forecast[asset] = result.forecast(
                    horizon=self.config.forecast_horizon
                ).variance.values[-1] * np.sqrt(252)
            
            return {
                'forecast': forecast
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze volatility forecast: {str(e)}")
            raise
    
    def _analyze_regime_volatility(self,
                                 returns: pd.DataFrame,
                                 regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze regime-based volatility."""
        try:
            if regime_manager is None:
                raise ValueError("Regime manager is required for regime-based volatility analysis")
            
            # Initialize results
            volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
            regime_volatility = {}
            
            # Detect current regime
            regime_result = regime_manager.detect_regime(returns.mean(axis=1))
            current_regime = regime_result['regime']
            
            # Calculate volatility by regime
            for regime in RegimeType:
                regime_returns = returns[
                    returns.index.isin(
                        returns.index[regime_result['regime'] == regime]
                    )
                ]
                regime_volatility[regime] = regime_returns.std() * np.sqrt(252)
            
            # Calculate current volatility
            for asset in returns.columns:
                volatility[asset] = regime_volatility[current_regime][asset]
            
            # Calculate volatility bands
            bands = {}
            for band in self.config.volatility_bands:
                bands[f"{band}std"] = volatility * band
            
            return {
                'regime': current_regime,
                'regime_volatility': regime_volatility,
                'volatility': volatility,
                'volatility_bands': bands
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze regime volatility: {str(e)}")
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
        
        # Initialize volatility manager
        config = VolatilityConfig(
            volatility_type=VolatilityType.GARCH,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            ewma_lambda=0.94,
            garch_order=(1, 1),
            realized_frequency="1D",
            forecast_horizon=10,
            regime_aware=True,
            regime_threshold=0.7,
            volatility_bands=[0.5, 1.0, 1.5, 2.0],
            significance_threshold=0.05
        )
        
        manager = VolatilityManager(config)
        
        # Analyze volatility
        result = manager.analyze_volatility(returns)
        
        # Print results
        print("\nVolatility Analysis Results:")
        print("\nVolatility:")
        print(result['volatility'])
        
        if 'parameters' in result:
            print("\nGARCH Parameters:")
            for asset, params in result['parameters'].items():
                print(f"\n{asset}:")
                for param, value in params.items():
                    print(f"  {param}: {value:.6f}")
        
        print("\nVolatility Bands:")
        for band, values in result['volatility_bands'].items():
            print(f"\n{band}:")
            print(values)
        
    except Exception as e:
        print(f"Error: {str(e)}") 
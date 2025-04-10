# ==========================
# risk/drawdown_manager.py
# ==========================
# Drawdown analysis and management.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from utils.logger import get_logger
from risk.market_regime import MarketRegimeManager, RegimeType
from risk.volatility_manager import VolatilityManager, VolatilityType

logger = get_logger("drawdown_manager")

class DrawdownType(Enum):
    """Types of drawdown analysis."""
    SIMPLE = "simple"        # Simple drawdown
    ROLLING = "rolling"      # Rolling drawdown
    REGIME = "regime"       # Regime-based drawdown
    CONDITIONAL = "conditional"  # Conditional drawdown
    STRESS = "stress"       # Stress test drawdown
    SCENARIO = "scenario"   # Scenario analysis drawdown
    HISTORICAL = "historical"  # Historical drawdown
    MONTE_CARLO = "monte_carlo"  # Monte Carlo drawdown
    BOOTSTRAP = "bootstrap"  # Bootstrap drawdown
    FORECAST = "forecast"   # Drawdown forecasting

@dataclass
class DrawdownConfig:
    """Configuration for drawdown analysis."""
    drawdown_type: DrawdownType = DrawdownType.SIMPLE
    lookback_window: int = 252  # Lookback window for drawdown calculation
    min_observations: int = 20  # Minimum observations required
    confidence_level: float = 0.95  # Confidence level for drawdown bands
    rolling_window: int = 60  # Window for rolling drawdown
    regime_aware: bool = True  # Whether to consider market regimes
    regime_threshold: float = 0.7  # Threshold for regime classification
    stress_periods: List[str] = None  # List of stress periods to analyze
    scenario_probabilities: Dict[str, float] = None  # Scenario probabilities
    monte_carlo_sims: int = 10000  # Number of Monte Carlo simulations
    bootstrap_samples: int = 1000  # Number of bootstrap samples
    forecast_horizon: int = 10  # Forecast horizon in days
    drawdown_thresholds: List[float] = None  # Custom drawdown thresholds
    significance_threshold: float = 0.05  # Significance threshold for tests

class DrawdownManager:
    """Manages drawdown analysis and management."""
    
    def __init__(self, config: DrawdownConfig):
        self.config = config
        if self.config.drawdown_thresholds is None:
            self.config.drawdown_thresholds = [-0.05, -0.10, -0.15, -0.20]
        if self.config.stress_periods is None:
            self.config.stress_periods = ["2008-09-15", "2020-03-16"]
        if self.config.scenario_probabilities is None:
            self.config.scenario_probabilities = {
                "normal": 0.7,
                "stress": 0.2,
                "crisis": 0.1
            }
        self._setup_drawdown_manager()
    
    def _setup_drawdown_manager(self) -> None:
        """Initialize drawdown manager."""
        try:
            logger.info("Initializing drawdown manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup drawdown manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate drawdown configuration."""
        try:
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.rolling_window < self.config.min_observations:
                raise ValueError("Rolling window must be at least min_observations")
            if self.config.regime_threshold <= 0 or self.config.regime_threshold >= 1:
                raise ValueError("Regime threshold must be between 0 and 1")
            if self.config.monte_carlo_sims < 1000:
                raise ValueError("Number of Monte Carlo simulations must be at least 1000")
            if self.config.bootstrap_samples < 100:
                raise ValueError("Number of bootstrap samples must be at least 100")
            if self.config.forecast_horizon < 1:
                raise ValueError("Forecast horizon must be at least 1")
            if self.config.significance_threshold <= 0 or self.config.significance_threshold >= 1:
                raise ValueError("Significance threshold must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid drawdown configuration: {str(e)}")
            raise
    
    def analyze_drawdown(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None,
                        volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """
        Analyze drawdowns of assets.
        
        Args:
            returns: DataFrame of asset returns
            regime_manager: Market regime manager (optional)
            volatility_manager: Volatility manager (optional)
            
        Returns:
            Dictionary containing drawdown analysis results
        """
        try:
            if self.config.drawdown_type == DrawdownType.SIMPLE:
                return self._analyze_simple_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.ROLLING:
                return self._analyze_rolling_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.REGIME:
                return self._analyze_regime_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.CONDITIONAL:
                return self._analyze_conditional_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.STRESS:
                return self._analyze_stress_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.SCENARIO:
                return self._analyze_scenario_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.HISTORICAL:
                return self._analyze_historical_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.MONTE_CARLO:
                return self._analyze_monte_carlo_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.BOOTSTRAP:
                return self._analyze_bootstrap_drawdown(returns, regime_manager, volatility_manager)
            elif self.config.drawdown_type == DrawdownType.FORECAST:
                return self._analyze_drawdown_forecast(returns, regime_manager, volatility_manager)
            else:
                raise ValueError(f"Unsupported drawdown type: {self.config.drawdown_type}")
        except Exception as e:
            logger.error(f"Failed to analyze drawdown: {str(e)}")
            raise
    
    def _analyze_simple_drawdown(self,
                               returns: pd.DataFrame,
                               regime_manager: Optional[MarketRegimeManager] = None,
                               volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze simple drawdown."""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cum_returns / running_max) - 1
            
            # Calculate maximum drawdown
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            duration = self._calculate_drawdown_duration(drawdown)
            
            # Calculate recovery time
            recovery = self._calculate_recovery_time(drawdown)
            
            return {
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'duration': duration,
                'recovery': recovery
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze simple drawdown: {str(e)}")
            raise
    
    def _analyze_rolling_drawdown(self,
                                returns: pd.DataFrame,
                                regime_manager: Optional[MarketRegimeManager] = None,
                                volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze rolling drawdown."""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate rolling maximum
            rolling_max = cum_returns.rolling(
                self.config.rolling_window
            ).max()
            
            # Calculate rolling drawdown
            drawdown = (cum_returns / rolling_max) - 1
            
            # Calculate maximum rolling drawdown
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            duration = self._calculate_drawdown_duration(drawdown)
            
            # Calculate recovery time
            recovery = self._calculate_recovery_time(drawdown)
            
            return {
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'duration': duration,
                'recovery': recovery
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze rolling drawdown: {str(e)}")
            raise
    
    def _analyze_regime_drawdown(self,
                               returns: pd.DataFrame,
                               regime_manager: Optional[MarketRegimeManager] = None,
                               volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze regime-based drawdown."""
        try:
            if regime_manager is None:
                raise ValueError("Regime manager is required for regime-based drawdown analysis")
            
            # Initialize results
            drawdown = pd.DataFrame(index=returns.index, columns=returns.columns)
            regime_drawdown = {}
            
            # Detect current regime
            regime_result = regime_manager.detect_regime(returns.mean(axis=1))
            current_regime = regime_result['regime']
            
            # Calculate drawdown by regime
            for regime in RegimeType:
                regime_returns = returns[
                    returns.index.isin(
                        returns.index[regime_result['regime'] == regime]
                    )
                ]
                regime_cum_returns = (1 + regime_returns).cumprod()
                regime_running_max = regime_cum_returns.expanding().max()
                regime_drawdown[regime] = (regime_cum_returns / regime_running_max) - 1
            
            # Calculate current drawdown
            for asset in returns.columns:
                drawdown[asset] = regime_drawdown[current_regime][asset]
            
            # Calculate maximum drawdown
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            duration = self._calculate_drawdown_duration(drawdown)
            
            # Calculate recovery time
            recovery = self._calculate_recovery_time(drawdown)
            
            return {
                'regime': current_regime,
                'regime_drawdown': regime_drawdown,
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'duration': duration,
                'recovery': recovery
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze regime drawdown: {str(e)}")
            raise
    
    def _analyze_conditional_drawdown(self,
                                    returns: pd.DataFrame,
                                    regime_manager: Optional[MarketRegimeManager] = None,
                                    volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze conditional drawdown."""
        try:
            if volatility_manager is None:
                raise ValueError("Volatility manager is required for conditional drawdown analysis")
            
            # Calculate volatility
            volatility_result = volatility_manager.analyze_volatility(returns)
            volatility = volatility_result['volatility']
            
            # Calculate conditional drawdown
            drawdown = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for asset in returns.columns:
                # Calculate drawdown conditional on volatility
                high_vol_returns = returns[asset][volatility[asset] > volatility[asset].quantile(0.9)]
                if len(high_vol_returns) > 0:
                    high_vol_cum_returns = (1 + high_vol_returns).cumprod()
                    high_vol_running_max = high_vol_cum_returns.expanding().max()
                    drawdown[asset] = (high_vol_cum_returns / high_vol_running_max) - 1
            
            # Calculate maximum drawdown
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            duration = self._calculate_drawdown_duration(drawdown)
            
            # Calculate recovery time
            recovery = self._calculate_recovery_time(drawdown)
            
            return {
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'duration': duration,
                'recovery': recovery
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze conditional drawdown: {str(e)}")
            raise
    
    def _analyze_stress_drawdown(self,
                               returns: pd.DataFrame,
                               regime_manager: Optional[MarketRegimeManager] = None,
                               volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze stress test drawdown."""
        try:
            # Initialize results
            stress_drawdown = {}
            
            # Calculate drawdown for each stress period
            for period in self.config.stress_periods:
                period_date = pd.to_datetime(period)
                stress_returns = returns[returns.index >= period_date]
                stress_cum_returns = (1 + stress_returns).cumprod()
                stress_running_max = stress_cum_returns.expanding().max()
                stress_drawdown[period] = (stress_cum_returns / stress_running_max) - 1
            
            return {
                'stress_drawdown': stress_drawdown
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze stress drawdown: {str(e)}")
            raise
    
    def _analyze_scenario_drawdown(self,
                                 returns: pd.DataFrame,
                                 regime_manager: Optional[MarketRegimeManager] = None,
                                 volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze scenario drawdown."""
        try:
            # Initialize results
            scenario_drawdown = {}
            
            # Calculate drawdown for each scenario
            for scenario, prob in self.config.scenario_probabilities.items():
                if scenario == "normal":
                    scenario_returns = returns
                elif scenario == "stress":
                    scenario_returns = returns * 1.5
                elif scenario == "crisis":
                    scenario_returns = returns * 2.0
                else:
                    continue
                
                scenario_cum_returns = (1 + scenario_returns).cumprod()
                scenario_running_max = scenario_cum_returns.expanding().max()
                scenario_drawdown[scenario] = (scenario_cum_returns / scenario_running_max) - 1
            
            return {
                'scenario_drawdown': scenario_drawdown
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze scenario drawdown: {str(e)}")
            raise
    
    def _analyze_historical_drawdown(self,
                                   returns: pd.DataFrame,
                                   regime_manager: Optional[MarketRegimeManager] = None,
                                   volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze historical drawdown."""
        try:
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate historical maximum
            historical_max = cum_returns.expanding().max()
            
            # Calculate historical drawdown
            drawdown = (cum_returns / historical_max) - 1
            
            # Calculate maximum historical drawdown
            max_drawdown = drawdown.min()
            
            # Calculate drawdown duration
            duration = self._calculate_drawdown_duration(drawdown)
            
            # Calculate recovery time
            recovery = self._calculate_recovery_time(drawdown)
            
            return {
                'drawdown': drawdown,
                'max_drawdown': max_drawdown,
                'duration': duration,
                'recovery': recovery
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze historical drawdown: {str(e)}")
            raise
    
    def _analyze_monte_carlo_drawdown(self,
                                    returns: pd.DataFrame,
                                    regime_manager: Optional[MarketRegimeManager] = None,
                                    volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze Monte Carlo drawdown."""
        try:
            # Initialize results
            simulated_drawdowns = []
            
            # Generate Monte Carlo simulations
            for _ in range(self.config.monte_carlo_sims):
                # Generate random returns
                simulated_returns = np.random.normal(
                    returns.mean(),
                    returns.std(),
                    size=len(returns)
                )
                simulated_returns = pd.DataFrame(
                    simulated_returns,
                    index=returns.index,
                    columns=returns.columns
                )
                
                # Calculate drawdown
                simulated_cum_returns = (1 + simulated_returns).cumprod()
                simulated_running_max = simulated_cum_returns.expanding().max()
                simulated_drawdown = (simulated_cum_returns / simulated_running_max) - 1
                
                simulated_drawdowns.append(simulated_drawdown)
            
            # Calculate statistics
            mean_drawdown = pd.concat(simulated_drawdowns, axis=1).mean(axis=1)
            std_drawdown = pd.concat(simulated_drawdowns, axis=1).std(axis=1)
            percentile_drawdown = pd.concat(simulated_drawdowns, axis=1).quantile(
                [0.05, 0.25, 0.75, 0.95],
                axis=1
            )
            
            return {
                'mean_drawdown': mean_drawdown,
                'std_drawdown': std_drawdown,
                'percentile_drawdown': percentile_drawdown
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze Monte Carlo drawdown: {str(e)}")
            raise
    
    def _analyze_bootstrap_drawdown(self,
                                  returns: pd.DataFrame,
                                  regime_manager: Optional[MarketRegimeManager] = None,
                                  volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze bootstrap drawdown."""
        try:
            # Initialize results
            bootstrap_drawdowns = []
            
            # Generate bootstrap samples
            for _ in range(self.config.bootstrap_samples):
                # Sample returns with replacement
                bootstrap_returns = returns.sample(
                    n=len(returns),
                    replace=True
                )
                
                # Calculate drawdown
                bootstrap_cum_returns = (1 + bootstrap_returns).cumprod()
                bootstrap_running_max = bootstrap_cum_returns.expanding().max()
                bootstrap_drawdown = (bootstrap_cum_returns / bootstrap_running_max) - 1
                
                bootstrap_drawdowns.append(bootstrap_drawdown)
            
            # Calculate statistics
            mean_drawdown = pd.concat(bootstrap_drawdowns, axis=1).mean(axis=1)
            std_drawdown = pd.concat(bootstrap_drawdowns, axis=1).std(axis=1)
            percentile_drawdown = pd.concat(bootstrap_drawdowns, axis=1).quantile(
                [0.05, 0.25, 0.75, 0.95],
                axis=1
            )
            
            return {
                'mean_drawdown': mean_drawdown,
                'std_drawdown': std_drawdown,
                'percentile_drawdown': percentile_drawdown
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze bootstrap drawdown: {str(e)}")
            raise
    
    def _analyze_drawdown_forecast(self,
                                 returns: pd.DataFrame,
                                 regime_manager: Optional[MarketRegimeManager] = None,
                                 volatility_manager: Optional[VolatilityManager] = None) -> Dict[str, Any]:
        """Analyze drawdown forecast."""
        try:
            if volatility_manager is None:
                raise ValueError("Volatility manager is required for drawdown forecasting")
            
            # Initialize results
            forecast = pd.DataFrame(index=range(self.config.forecast_horizon), columns=returns.columns)
            
            # Forecast drawdown for each asset
            for asset in returns.columns:
                # Fit GARCH model
                model = arch_model(
                    returns[asset],
                    p=1,
                    q=1
                )
                result = model.fit(disp='off')
                
                # Generate return forecasts
                return_forecast = result.forecast(
                    horizon=self.config.forecast_horizon
                ).mean.values[-1]
                
                # Calculate drawdown forecast
                cum_returns = (1 + return_forecast).cumprod()
                running_max = cum_returns.expanding().max()
                forecast[asset] = (cum_returns / running_max) - 1
            
            return {
                'forecast': forecast
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze drawdown forecast: {str(e)}")
            raise
    
    def _calculate_drawdown_duration(self, drawdown: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown duration."""
        try:
            duration = pd.DataFrame(index=drawdown.index, columns=drawdown.columns)
            
            for asset in drawdown.columns:
                in_drawdown = drawdown[asset] < 0
                duration[asset] = in_drawdown.groupby((~in_drawdown).cumsum()).cumsum()
            
            return duration
            
        except Exception as e:
            logger.error(f"Failed to calculate drawdown duration: {str(e)}")
            raise
    
    def _calculate_recovery_time(self, drawdown: pd.DataFrame) -> pd.DataFrame:
        """Calculate recovery time."""
        try:
            recovery = pd.DataFrame(index=drawdown.index, columns=drawdown.columns)
            
            for asset in drawdown.columns:
                in_drawdown = drawdown[asset] < 0
                recovery[asset] = in_drawdown.groupby((~in_drawdown).cumsum()).cumcount()
            
            return recovery
            
        except Exception as e:
            logger.error(f"Failed to calculate recovery time: {str(e)}")
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
        
        # Initialize drawdown manager
        config = DrawdownConfig(
            drawdown_type=DrawdownType.SIMPLE,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            rolling_window=60,
            regime_aware=True,
            regime_threshold=0.7,
            stress_periods=["2008-09-15", "2020-03-16"],
            scenario_probabilities={
                "normal": 0.7,
                "stress": 0.2,
                "crisis": 0.1
            },
            monte_carlo_sims=10000,
            bootstrap_samples=1000,
            forecast_horizon=10,
            drawdown_thresholds=[-0.05, -0.10, -0.15, -0.20],
            significance_threshold=0.05
        )
        
        manager = DrawdownManager(config)
        
        # Analyze drawdown
        result = manager.analyze_drawdown(returns)
        
        # Print results
        print("\nDrawdown Analysis Results:")
        print("\nDrawdown:")
        print(result['drawdown'])
        
        print("\nMaximum Drawdown:")
        print(result['max_drawdown'])
        
        print("\nDrawdown Duration:")
        print(result['duration'])
        
        print("\nRecovery Time:")
        print(result['recovery'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 
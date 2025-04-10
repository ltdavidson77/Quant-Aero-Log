# ==========================
# risk/risk_metrics.py
# ==========================
# Comprehensive risk metrics calculations.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from utils.logger import get_logger

logger = get_logger("risk_metrics")

class RiskMetricType(Enum):
    """Types of risk metrics."""
    VAR = "var"                  # Value at Risk
    CVAR = "cvar"                # Conditional VaR
    EXPECTED_SHORTFALL = "es"    # Expected Shortfall
    SHARPE_RATIO = "sharpe"      # Sharpe Ratio
    SORTINO_RATIO = "sortino"    # Sortino Ratio
    MAX_DRAWDOWN = "drawdown"    # Maximum Drawdown
    VOLATILITY = "volatility"    # Volatility
    BETA = "beta"                # Beta
    CORRELATION = "correlation"  # Correlation
    STRESS_TEST = "stress"       # Stress Test
    SCENARIO = "scenario"        # Scenario Analysis

@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculations."""
    var_confidence: float = 0.95  # VaR confidence level
    cvar_confidence: float = 0.99  # CVaR confidence level
    risk_free_rate: float = 0.02  # Risk-free rate
    lookback_window: int = 252  # Lookback window for calculations
    monte_carlo_sims: int = 10000  # Number of Monte Carlo simulations
    stress_test_scenarios: List[Dict[str, float]] = None  # Stress test scenarios
    scenario_analysis: List[Dict[str, float]] = None  # Scenario analysis parameters
    correlation_window: int = 60  # Window for correlation calculation
    beta_window: int = 252  # Window for beta calculation
    volatility_window: int = 20  # Window for volatility calculation
    drawdown_window: int = 252  # Window for drawdown calculation
    min_observations: int = 20  # Minimum observations required
    bootstrap_samples: int = 1000  # Number of bootstrap samples
    confidence_interval: float = 0.95  # Confidence interval for estimates

class RiskMetricsManager:
    """Manages risk metrics calculations."""
    
    def __init__(self, config: RiskMetricsConfig):
        self.config = config
        self._setup_risk_metrics()
    
    def _setup_risk_metrics(self) -> None:
        """Initialize risk metrics manager."""
        try:
            logger.info("Initializing risk metrics manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup risk metrics manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate risk metrics configuration."""
        try:
            if self.config.var_confidence <= 0 or self.config.var_confidence >= 1:
                raise ValueError("VaR confidence must be between 0 and 1")
            if self.config.cvar_confidence <= 0 or self.config.cvar_confidence >= 1:
                raise ValueError("CVaR confidence must be between 0 and 1")
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.monte_carlo_sims < 1000:
                raise ValueError("Monte Carlo simulations must be at least 1000")
            if self.config.correlation_window < 2:
                raise ValueError("Correlation window must be at least 2")
            if self.config.beta_window < 2:
                raise ValueError("Beta window must be at least 2")
            if self.config.volatility_window < 2:
                raise ValueError("Volatility window must be at least 2")
            if self.config.drawdown_window < 2:
                raise ValueError("Drawdown window must be at least 2")
            if self.config.bootstrap_samples < 100:
                raise ValueError("Bootstrap samples must be at least 100")
            if self.config.confidence_interval <= 0 or self.config.confidence_interval >= 1:
                raise ValueError("Confidence interval must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid risk metrics configuration: {str(e)}")
            raise
    
    def calculate_risk_metrics(self,
                             returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None,
                             metric_types: List[RiskMetricType] = None) -> Dict[str, float]:
        """
        Calculate risk metrics for given returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns (optional)
            metric_types: List of risk metrics to calculate
            
        Returns:
            Dictionary of calculated risk metrics
        """
        try:
            if metric_types is None:
                metric_types = list(RiskMetricType)
            
            metrics = {}
            
            for metric_type in metric_types:
                if metric_type == RiskMetricType.VAR:
                    metrics['var'] = self._calculate_var(returns)
                elif metric_type == RiskMetricType.CVAR:
                    metrics['cvar'] = self._calculate_cvar(returns)
                elif metric_type == RiskMetricType.EXPECTED_SHORTFALL:
                    metrics['expected_shortfall'] = self._calculate_expected_shortfall(returns)
                elif metric_type == RiskMetricType.SHARPE_RATIO:
                    metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
                elif metric_type == RiskMetricType.SORTINO_RATIO:
                    metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                elif metric_type == RiskMetricType.MAX_DRAWDOWN:
                    metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
                elif metric_type == RiskMetricType.VOLATILITY:
                    metrics['volatility'] = self._calculate_volatility(returns)
                elif metric_type == RiskMetricType.BETA:
                    if benchmark_returns is not None:
                        metrics['beta'] = self._calculate_beta(returns, benchmark_returns)
                elif metric_type == RiskMetricType.CORRELATION:
                    if benchmark_returns is not None:
                        metrics['correlation'] = self._calculate_correlation(returns, benchmark_returns)
                elif metric_type == RiskMetricType.STRESS_TEST:
                    metrics['stress_test'] = self._calculate_stress_test(returns)
                elif metric_type == RiskMetricType.SCENARIO:
                    metrics['scenario_analysis'] = self._calculate_scenario_analysis(returns)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {str(e)}")
            raise
    
    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk."""
        try:
            # Calculate historical VaR
            var = np.percentile(returns, (1 - self.config.var_confidence) * 100)
            
            # Calculate confidence interval using bootstrap
            bootstrap_vars = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_vars.append(np.percentile(sample, (1 - self.config.var_confidence) * 100))
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_vars, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_vars, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"VaR: {var:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return var
            
        except Exception as e:
            logger.error(f"Failed to calculate VaR: {str(e)}")
            raise
    
    def _calculate_cvar(self, returns: pd.Series) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            # Calculate VaR
            var = self._calculate_var(returns)
            
            # Calculate CVaR
            cvar = returns[returns <= var].mean()
            
            # Calculate confidence interval using bootstrap
            bootstrap_cvars = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(returns, size=len(returns), replace=True)
                sample_var = np.percentile(sample, (1 - self.config.cvar_confidence) * 100)
                bootstrap_cvars.append(sample[sample <= sample_var].mean())
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_cvars, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_cvars, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"CVaR: {cvar:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return cvar
            
        except Exception as e:
            logger.error(f"Failed to calculate CVaR: {str(e)}")
            raise
    
    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall."""
        try:
            # Calculate VaR
            var = self._calculate_var(returns)
            
            # Calculate Expected Shortfall
            es = returns[returns <= var].mean()
            
            # Calculate confidence interval using bootstrap
            bootstrap_es = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(returns, size=len(returns), replace=True)
                sample_var = np.percentile(sample, (1 - self.config.var_confidence) * 100)
                bootstrap_es.append(sample[sample <= sample_var].mean())
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_es, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_es, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Expected Shortfall: {es:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return es
            
        except Exception as e:
            logger.error(f"Failed to calculate Expected Shortfall: {str(e)}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio."""
        try:
            # Calculate excess returns
            excess_returns = returns - self.config.risk_free_rate
            
            # Calculate Sharpe Ratio
            sharpe = excess_returns.mean() / excess_returns.std()
            
            # Calculate confidence interval using bootstrap
            bootstrap_sharpes = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(excess_returns, size=len(excess_returns), replace=True)
                bootstrap_sharpes.append(sample.mean() / sample.std())
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_sharpes, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_sharpes, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Sharpe Ratio: {sharpe:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return sharpe
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe Ratio: {str(e)}")
            raise
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino Ratio."""
        try:
            # Calculate excess returns
            excess_returns = returns - self.config.risk_free_rate
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.sqrt((downside_returns ** 2).mean())
            
            # Calculate Sortino Ratio
            sortino = excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0
            
            # Calculate confidence interval using bootstrap
            bootstrap_sortinos = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(excess_returns, size=len(excess_returns), replace=True)
                sample_downside = sample[sample < 0]
                sample_deviation = np.sqrt((sample_downside ** 2).mean())
                bootstrap_sortinos.append(sample.mean() / sample_deviation if sample_deviation != 0 else 0)
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_sortinos, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_sortinos, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Sortino Ratio: {sortino:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return sortino
            
        except Exception as e:
            logger.error(f"Failed to calculate Sortino Ratio: {str(e)}")
            raise
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdowns
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Calculate maximum drawdown
            max_drawdown = drawdowns.min()
            
            # Calculate confidence interval using bootstrap
            bootstrap_drawdowns = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(returns, size=len(returns), replace=True)
                sample_cumulative = (1 + sample).cumprod()
                sample_max = sample_cumulative.expanding().max()
                sample_drawdowns = (sample_cumulative - sample_max) / sample_max
                bootstrap_drawdowns.append(sample_drawdowns.min())
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_drawdowns, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_drawdowns, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Maximum Drawdown: {max_drawdown:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Failed to calculate Maximum Drawdown: {str(e)}")
            raise
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate Volatility."""
        try:
            # Calculate volatility
            volatility = returns.rolling(window=self.config.volatility_window).std().iloc[-1]
            
            # Calculate confidence interval using bootstrap
            bootstrap_volatilities = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_volatilities.append(sample.rolling(window=self.config.volatility_window).std().iloc[-1])
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_volatilities, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_volatilities, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Volatility: {volatility:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return volatility
            
        except Exception as e:
            logger.error(f"Failed to calculate Volatility: {str(e)}")
            raise
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Beta."""
        try:
            # Calculate covariance and variance
            covariance = returns.rolling(window=self.config.beta_window).cov(benchmark_returns)
            variance = benchmark_returns.rolling(window=self.config.beta_window).var()
            
            # Calculate beta
            beta = covariance.iloc[-1] / variance.iloc[-1] if variance.iloc[-1] != 0 else 0
            
            # Calculate confidence interval using bootstrap
            bootstrap_betas = []
            for _ in range(self.config.bootstrap_samples):
                sample_idx = np.random.choice(len(returns), size=len(returns), replace=True)
                sample_returns = returns.iloc[sample_idx]
                sample_benchmark = benchmark_returns.iloc[sample_idx]
                sample_cov = sample_returns.rolling(window=self.config.beta_window).cov(sample_benchmark)
                sample_var = sample_benchmark.rolling(window=self.config.beta_window).var()
                bootstrap_betas.append(sample_cov.iloc[-1] / sample_var.iloc[-1] if sample_var.iloc[-1] != 0 else 0)
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_betas, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_betas, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Beta: {beta:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return beta
            
        except Exception as e:
            logger.error(f"Failed to calculate Beta: {str(e)}")
            raise
    
    def _calculate_correlation(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Correlation."""
        try:
            # Calculate correlation
            correlation = returns.rolling(window=self.config.correlation_window).corr(benchmark_returns).iloc[-1]
            
            # Calculate confidence interval using bootstrap
            bootstrap_correlations = []
            for _ in range(self.config.bootstrap_samples):
                sample_idx = np.random.choice(len(returns), size=len(returns), replace=True)
                sample_returns = returns.iloc[sample_idx]
                sample_benchmark = benchmark_returns.iloc[sample_idx]
                bootstrap_correlations.append(
                    sample_returns.rolling(window=self.config.correlation_window).corr(sample_benchmark).iloc[-1]
                )
            
            # Calculate confidence interval
            lower_bound = np.percentile(bootstrap_correlations, (1 - self.config.confidence_interval) * 50)
            upper_bound = np.percentile(bootstrap_correlations, (1 + self.config.confidence_interval) * 50)
            
            logger.info(f"Correlation: {correlation:.4f} (CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
            return correlation
            
        except Exception as e:
            logger.error(f"Failed to calculate Correlation: {str(e)}")
            raise
    
    def _calculate_stress_test(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Stress Test results."""
        try:
            stress_results = {}
            
            if self.config.stress_test_scenarios:
                for scenario in self.config.stress_test_scenarios:
                    # Apply stress scenario
                    stressed_returns = returns.copy()
                    for factor, shock in scenario.items():
                        stressed_returns *= (1 + shock)
                    
                    # Calculate metrics under stress
                    stress_results[scenario['name']] = {
                        'var': self._calculate_var(stressed_returns),
                        'cvar': self._calculate_cvar(stressed_returns),
                        'max_drawdown': self._calculate_max_drawdown(stressed_returns)
                    }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Failed to calculate Stress Test: {str(e)}")
            raise
    
    def _calculate_scenario_analysis(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Scenario Analysis results."""
        try:
            scenario_results = {}
            
            if self.config.scenario_analysis:
                for scenario in self.config.scenario_analysis:
                    # Apply scenario
                    scenario_returns = returns.copy()
                    for factor, change in scenario.items():
                        scenario_returns *= (1 + change)
                    
                    # Calculate metrics under scenario
                    scenario_results[scenario['name']] = {
                        'expected_return': scenario_returns.mean(),
                        'volatility': self._calculate_volatility(scenario_returns),
                        'sharpe_ratio': self._calculate_sharpe_ratio(scenario_returns)
                    }
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Failed to calculate Scenario Analysis: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.015, 252), index=dates)
        
        # Define stress test scenarios
        stress_scenarios = [
            {'name': 'market_crash', 'shock': -0.2},
            {'name': 'volatility_spike', 'shock': 0.1},
            {'name': 'liquidity_crisis', 'shock': -0.15}
        ]
        
        # Define scenario analysis
        scenario_analysis = [
            {'name': 'bull_market', 'change': 0.1},
            {'name': 'bear_market', 'change': -0.1},
            {'name': 'high_volatility', 'change': 0.05}
        ]
        
        # Initialize risk metrics manager
        config = RiskMetricsConfig(
            var_confidence=0.95,
            cvar_confidence=0.99,
            risk_free_rate=0.02,
            lookback_window=252,
            monte_carlo_sims=10000,
            stress_test_scenarios=stress_scenarios,
            scenario_analysis=scenario_analysis,
            correlation_window=60,
            beta_window=252,
            volatility_window=20,
            drawdown_window=252,
            min_observations=20,
            bootstrap_samples=1000,
            confidence_interval=0.95
        )
        
        manager = RiskMetricsManager(config)
        
        # Calculate risk metrics
        metrics = manager.calculate_risk_metrics(
            returns,
            benchmark_returns,
            metric_types=[
                RiskMetricType.VAR,
                RiskMetricType.CVAR,
                RiskMetricType.EXPECTED_SHORTFALL,
                RiskMetricType.SHARPE_RATIO,
                RiskMetricType.SORTINO_RATIO,
                RiskMetricType.MAX_DRAWDOWN,
                RiskMetricType.VOLATILITY,
                RiskMetricType.BETA,
                RiskMetricType.CORRELATION,
                RiskMetricType.STRESS_TEST,
                RiskMetricType.SCENARIO
            ]
        )
        
        # Print results
        print("\nRisk Metrics Results:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{metric}:")
                for scenario, results in value.items():
                    print(f"  {scenario}:")
                    for k, v in results.items():
                        print(f"    {k}: {v:.4f}")
            else:
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 
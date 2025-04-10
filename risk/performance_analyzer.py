# ==========================
# risk/performance_analyzer.py
# ==========================
# Performance analysis and attribution.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager, RiskMetricType
from risk.portfolio_optimizer import PortfolioOptimizer, OptimizationType
from risk.risk_allocator import RiskAllocator, AllocationType

logger = get_logger("performance_analyzer")

class AttributionType(Enum):
    """Types of performance attribution."""
    BRINSON = "brinson"        # Brinson-Fachler attribution
    CARINO = "carino"          # Carino attribution
    MENCHER = "menchero"       # Menchero attribution
    ANKER = "anker"           # Anker attribution
    GEOMETRIC = "geometric"    # Geometric attribution
    ARITHMETIC = "arithmetic"  # Arithmetic attribution
    CUSTOM = "custom"         # Custom attribution

@dataclass
class PerformanceConfig:
    """Configuration for performance analysis."""
    attribution_type: AttributionType = AttributionType.BRINSON
    benchmark: str = None  # Benchmark identifier
    risk_free_rate: float = 0.02  # Risk-free rate
    lookback_window: int = 252  # Lookback window
    min_observations: int = 20  # Minimum observations
    confidence_level: float = 0.95  # Confidence level
    attribution_factors: List[str] = None  # Attribution factors
    custom_weights: Dict[str, float] = None  # Custom factor weights
    transaction_costs: float = 0.001  # Transaction costs
    tax_rate: float = 0.2  # Tax rate
    rebalancing_frequency: str = "M"  # Rebalancing frequency
    attribution_threshold: float = 0.01  # Attribution threshold

class PerformanceAnalyzer:
    """Manages performance analysis and attribution."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        if self.config.attribution_factors is None:
            self.config.attribution_factors = []
        if self.config.custom_weights is None:
            self.config.custom_weights = {}
        self._setup_performance_analyzer()
    
    def _setup_performance_analyzer(self) -> None:
        """Initialize performance analyzer."""
        try:
            logger.info("Initializing performance analyzer")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup performance analyzer: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate performance configuration."""
        try:
            if self.config.risk_free_rate < 0:
                raise ValueError("Risk-free rate must be non-negative")
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.transaction_costs < 0:
                raise ValueError("Transaction costs must be non-negative")
            if self.config.tax_rate < 0 or self.config.tax_rate >= 1:
                raise ValueError("Tax rate must be between 0 and 1")
            if self.config.attribution_threshold <= 0:
                raise ValueError("Attribution threshold must be positive")
        except Exception as e:
            logger.error(f"Invalid performance configuration: {str(e)}")
            raise
    
    def analyze_performance(self,
                          returns: pd.DataFrame,
                          weights: pd.DataFrame,
                          benchmark_returns: Optional[pd.Series] = None,
                          risk_metrics_manager: Optional[RiskMetricsManager] = None,
                          portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                          risk_allocator: Optional[RiskAllocator] = None) -> Dict[str, Any]:
        """
        Analyze portfolio performance.
        
        Args:
            returns: DataFrame of asset returns
            weights: DataFrame of portfolio weights
            benchmark_returns: Series of benchmark returns (optional)
            risk_metrics_manager: Risk metrics manager (optional)
            portfolio_optimizer: Portfolio optimizer (optional)
            risk_allocator: Risk allocator (optional)
            
        Returns:
            Dictionary containing performance analysis results
        """
        try:
            # Calculate basic performance metrics
            performance_metrics = self._calculate_performance_metrics(returns, weights, benchmark_returns)
            
            # Calculate attribution
            attribution = self._calculate_attribution(returns, weights, benchmark_returns)
            
            # Calculate risk-adjusted returns
            risk_adjusted = self._calculate_risk_adjusted_returns(returns, weights, risk_metrics_manager)
            
            # Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(weights)
            
            # Calculate tax impact
            tax_impact = self._calculate_tax_impact(returns, weights)
            
            # Calculate factor attribution
            factor_attribution = self._calculate_factor_attribution(returns, weights, benchmark_returns)
            
            # Calculate sector attribution
            sector_attribution = self._calculate_sector_attribution(returns, weights, benchmark_returns)
            
            # Calculate style attribution
            style_attribution = self._calculate_style_attribution(returns, weights, benchmark_returns)
            
            # Calculate region attribution
            region_attribution = self._calculate_region_attribution(returns, weights, benchmark_returns)
            
            return {
                'performance_metrics': performance_metrics,
                'attribution': attribution,
                'risk_adjusted': risk_adjusted,
                'transaction_costs': transaction_costs,
                'tax_impact': tax_impact,
                'factor_attribution': factor_attribution,
                'sector_attribution': sector_attribution,
                'style_attribution': style_attribution,
                'region_attribution': region_attribution
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self,
                                    returns: pd.DataFrame,
                                    weights: pd.DataFrame,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Calculate metrics
            metrics = {
                'total_return': portfolio_returns.sum(),
                'annualized_return': (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': (portfolio_returns.mean() - self.config.risk_free_rate) / portfolio_returns.std() * np.sqrt(252),
                'sortino_ratio': (portfolio_returns.mean() - self.config.risk_free_rate) / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252),
                'omega_ratio': self._calculate_omega_ratio(portfolio_returns),
                'sterling_ratio': self._calculate_sterling_ratio(portfolio_returns),
                'calmar_ratio': self._calculate_calmar_ratio(portfolio_returns),
                'max_drawdown': portfolio_returns.cumsum().min(),
                'win_rate': (portfolio_returns > 0).mean(),
                'avg_win': portfolio_returns[portfolio_returns > 0].mean(),
                'avg_loss': portfolio_returns[portfolio_returns < 0].mean(),
                'profit_factor': abs(portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()),
                'recovery_factor': abs(portfolio_returns.sum() / portfolio_returns.cumsum().min()),
                'tail_ratio': self._calculate_tail_ratio(portfolio_returns),
                'value_at_risk': self._calculate_value_at_risk(portfolio_returns),
                'expected_shortfall': self._calculate_expected_shortfall(portfolio_returns)
            }
            
            # Calculate benchmark metrics if available
            if benchmark_returns is not None:
                metrics.update({
                    'excess_return': metrics['annualized_return'] - benchmark_returns.mean() * 252,
                    'tracking_error': (portfolio_returns - benchmark_returns).std() * np.sqrt(252),
                    'information_ratio': metrics['excess_return'] / metrics['tracking_error'],
                    'beta': np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns),
                    'alpha': metrics['annualized_return'] - (self.config.risk_free_rate + metrics['beta'] * (benchmark_returns.mean() * 252 - self.config.risk_free_rate)),
                    'treynor_ratio': (metrics['annualized_return'] - self.config.risk_free_rate) / metrics['beta'],
                    'jensen_alpha': metrics['alpha'],
                    'm2_measure': self._calculate_m2_measure(portfolio_returns, benchmark_returns),
                    'appraisal_ratio': metrics['alpha'] / metrics['tracking_error']
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {str(e)}")
            raise
    
    def _calculate_attribution(self,
                            returns: pd.DataFrame,
                            weights: pd.DataFrame,
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate performance attribution."""
        try:
            if self.config.attribution_type == AttributionType.BRINSON:
                return self._calculate_brinson_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.CARINO:
                return self._calculate_carino_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.MENCHER:
                return self._calculate_menchero_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.ANKER:
                return self._calculate_anker_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.GEOMETRIC:
                return self._calculate_geometric_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.ARITHMETIC:
                return self._calculate_arithmetic_attribution(returns, weights, benchmark_returns)
            elif self.config.attribution_type == AttributionType.CUSTOM:
                return self._calculate_custom_attribution(returns, weights, benchmark_returns)
            else:
                raise ValueError(f"Unsupported attribution type: {self.config.attribution_type}")
                
        except Exception as e:
            logger.error(f"Failed to calculate attribution: {str(e)}")
            raise
    
    def _calculate_brinson_attribution(self,
                                    returns: pd.DataFrame,
                                    weights: pd.DataFrame,
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate Brinson-Fachler attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for Brinson attribution")
            
            # Calculate allocation effect
            allocation = (weights - weights.mean()) * (returns.mean() - returns.mean().mean())
            
            # Calculate selection effect
            selection = weights.mean() * (returns - returns.mean())
            
            # Calculate interaction effect
            interaction = (weights - weights.mean()) * (returns - returns.mean())
            
            return {
                'allocation': allocation.sum(),
                'selection': selection.sum(),
                'interaction': interaction.sum(),
                'total': allocation.sum() + selection.sum() + interaction.sum()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate Brinson attribution: {str(e)}")
            raise
    
    def _calculate_carino_attribution(self,
                                   returns: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate Carino attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for Carino attribution")
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = benchmark_returns
            
            # Calculate Carino factors
            k = np.log(1 + portfolio_returns.sum()) / portfolio_returns.sum()
            
            # Calculate attribution
            attribution = {
                'allocation': k * (weights - weights.mean()) * (returns.mean() - returns.mean().mean()),
                'selection': k * weights.mean() * (returns - returns.mean()),
                'interaction': k * (weights - weights.mean()) * (returns - returns.mean())
            }
            
            return {
                'allocation': attribution['allocation'].sum(),
                'selection': attribution['selection'].sum(),
                'interaction': attribution['interaction'].sum(),
                'total': attribution['allocation'].sum() + attribution['selection'].sum() + attribution['interaction'].sum()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate Carino attribution: {str(e)}")
            raise
    
    def _calculate_menchero_attribution(self,
                                     returns: pd.DataFrame,
                                     weights: pd.DataFrame,
                                     benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate Menchero attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for Menchero attribution")
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = benchmark_returns
            
            # Calculate Menchero factors
            k = np.log(1 + portfolio_returns.sum()) / portfolio_returns.sum()
            k_benchmark = np.log(1 + benchmark_returns.sum()) / benchmark_returns.sum()
            
            # Calculate attribution
            attribution = {
                'allocation': k * (weights - weights.mean()) * (returns.mean() - returns.mean().mean()),
                'selection': k * weights.mean() * (returns - returns.mean()),
                'interaction': k * (weights - weights.mean()) * (returns - returns.mean())
            }
            
            return {
                'allocation': attribution['allocation'].sum(),
                'selection': attribution['selection'].sum(),
                'interaction': attribution['interaction'].sum(),
                'total': attribution['allocation'].sum() + attribution['selection'].sum() + attribution['interaction'].sum()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate Menchero attribution: {str(e)}")
            raise
    
    def _calculate_anker_attribution(self,
                                  returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate Anker attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for Anker attribution")
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = benchmark_returns
            
            # Calculate Anker factors
            k = np.log(1 + portfolio_returns.sum()) / portfolio_returns.sum()
            k_benchmark = np.log(1 + benchmark_returns.sum()) / benchmark_returns.sum()
            
            # Calculate attribution
            attribution = {
                'allocation': k * (weights - weights.mean()) * (returns.mean() - returns.mean().mean()),
                'selection': k * weights.mean() * (returns - returns.mean()),
                'interaction': k * (weights - weights.mean()) * (returns - returns.mean())
            }
            
            return {
                'allocation': attribution['allocation'].sum(),
                'selection': attribution['selection'].sum(),
                'interaction': attribution['interaction'].sum(),
                'total': attribution['allocation'].sum() + attribution['selection'].sum() + attribution['interaction'].sum()
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate Anker attribution: {str(e)}")
            raise
    
    def _calculate_geometric_attribution(self,
                                      returns: pd.DataFrame,
                                      weights: pd.DataFrame,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate geometric attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for geometric attribution")
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = benchmark_returns
            
            # Calculate geometric attribution
            attribution = {
                'allocation': (1 + portfolio_returns).prod() / (1 + benchmark_returns).prod() - 1,
                'selection': (1 + portfolio_returns).prod() / (1 + benchmark_returns).prod() - 1,
                'interaction': (1 + portfolio_returns).prod() / (1 + benchmark_returns).prod() - 1
            }
            
            return {
                'allocation': attribution['allocation'],
                'selection': attribution['selection'],
                'interaction': attribution['interaction'],
                'total': attribution['allocation'] + attribution['selection'] + attribution['interaction']
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate geometric attribution: {str(e)}")
            raise
    
    def _calculate_arithmetic_attribution(self,
                                       returns: pd.DataFrame,
                                       weights: pd.DataFrame,
                                       benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate arithmetic attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for arithmetic attribution")
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = benchmark_returns
            
            # Calculate arithmetic attribution
            attribution = {
                'allocation': portfolio_returns.mean() - benchmark_returns.mean(),
                'selection': portfolio_returns.mean() - benchmark_returns.mean(),
                'interaction': portfolio_returns.mean() - benchmark_returns.mean()
            }
            
            return {
                'allocation': attribution['allocation'],
                'selection': attribution['selection'],
                'interaction': attribution['interaction'],
                'total': attribution['allocation'] + attribution['selection'] + attribution['interaction']
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate arithmetic attribution: {str(e)}")
            raise
    
    def _calculate_factor_attribution(self,
                                   returns: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate factor attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for factor attribution")
            
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(returns, weights)
            
            # Calculate factor returns
            factor_returns = self._calculate_factor_returns(returns, weights)
            
            # Calculate factor attribution
            attribution = {
                'market': factor_returns['market'] * factor_exposures['market'],
                'size': factor_returns['size'] * factor_exposures['size'],
                'value': factor_returns['value'] * factor_exposures['value'],
                'momentum': factor_returns['momentum'] * factor_exposures['momentum'],
                'quality': factor_returns['quality'] * factor_exposures['quality'],
                'volatility': factor_returns['volatility'] * factor_exposures['volatility']
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Failed to calculate factor attribution: {str(e)}")
            raise
    
    def _calculate_sector_attribution(self,
                                   returns: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate sector attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for sector attribution")
            
            # Calculate sector exposures
            sector_exposures = self._calculate_sector_exposures(returns, weights)
            
            # Calculate sector returns
            sector_returns = self._calculate_sector_returns(returns, weights)
            
            # Calculate sector attribution
            attribution = {
                'technology': sector_returns['technology'] * sector_exposures['technology'],
                'financials': sector_returns['financials'] * sector_exposures['financials'],
                'healthcare': sector_returns['healthcare'] * sector_exposures['healthcare'],
                'consumer': sector_returns['consumer'] * sector_exposures['consumer'],
                'industrials': sector_returns['industrials'] * sector_exposures['industrials'],
                'energy': sector_returns['energy'] * sector_exposures['energy'],
                'materials': sector_returns['materials'] * sector_exposures['materials'],
                'utilities': sector_returns['utilities'] * sector_exposures['utilities'],
                'real_estate': sector_returns['real_estate'] * sector_exposures['real_estate'],
                'telecom': sector_returns['telecom'] * sector_exposures['telecom']
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Failed to calculate sector attribution: {str(e)}")
            raise
    
    def _calculate_style_attribution(self,
                                  returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate style attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for style attribution")
            
            # Calculate style exposures
            style_exposures = self._calculate_style_exposures(returns, weights)
            
            # Calculate style returns
            style_returns = self._calculate_style_returns(returns, weights)
            
            # Calculate style attribution
            attribution = {
                'growth': style_returns['growth'] * style_exposures['growth'],
                'value': style_returns['value'] * style_exposures['value'],
                'momentum': style_returns['momentum'] * style_exposures['momentum'],
                'quality': style_returns['quality'] * style_exposures['quality'],
                'size': style_returns['size'] * style_exposures['size'],
                'volatility': style_returns['volatility'] * style_exposures['volatility']
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Failed to calculate style attribution: {str(e)}")
            raise
    
    def _calculate_region_attribution(self,
                                   returns: pd.DataFrame,
                                   weights: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate region attribution."""
        try:
            if benchmark_returns is None:
                raise ValueError("Benchmark returns are required for region attribution")
            
            # Calculate region exposures
            region_exposures = self._calculate_region_exposures(returns, weights)
            
            # Calculate region returns
            region_returns = self._calculate_region_returns(returns, weights)
            
            # Calculate region attribution
            attribution = {
                'north_america': region_returns['north_america'] * region_exposures['north_america'],
                'europe': region_returns['europe'] * region_exposures['europe'],
                'asia_pacific': region_returns['asia_pacific'] * region_exposures['asia_pacific'],
                'emerging_markets': region_returns['emerging_markets'] * region_exposures['emerging_markets'],
                'latin_america': region_returns['latin_america'] * region_exposures['latin_america'],
                'middle_east': region_returns['middle_east'] * region_exposures['middle_east']
            }
            
            return attribution
            
        except Exception as e:
            logger.error(f"Failed to calculate region attribution: {str(e)}")
            raise
    
    def _calculate_omega_ratio(self, returns: pd.Series) -> float:
        """Calculate Omega ratio."""
        try:
            threshold = self.config.risk_free_rate
            positive_returns = returns[returns > threshold]
            negative_returns = returns[returns <= threshold]
            return positive_returns.sum() / abs(negative_returns.sum())
        except Exception as e:
            logger.error(f"Failed to calculate Omega ratio: {str(e)}")
            raise
    
    def _calculate_sterling_ratio(self, returns: pd.Series) -> float:
        """Calculate Sterling ratio."""
        try:
            max_drawdown = returns.cumsum().min()
            return (returns.mean() - self.config.risk_free_rate) / abs(max_drawdown)
        except Exception as e:
            logger.error(f"Failed to calculate Sterling ratio: {str(e)}")
            raise
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        try:
            max_drawdown = returns.cumsum().min()
            return (returns.mean() - self.config.risk_free_rate) / abs(max_drawdown)
        except Exception as e:
            logger.error(f"Failed to calculate Calmar ratio: {str(e)}")
            raise
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        try:
            positive_tail = returns[returns > 0].quantile(0.95)
            negative_tail = returns[returns < 0].quantile(0.05)
            return abs(positive_tail / negative_tail)
        except Exception as e:
            logger.error(f"Failed to calculate tail ratio: {str(e)}")
            raise
    
    def _calculate_value_at_risk(self, returns: pd.Series) -> float:
        """Calculate Value at Risk."""
        try:
            return returns.quantile(1 - self.config.confidence_level)
        except Exception as e:
            logger.error(f"Failed to calculate Value at Risk: {str(e)}")
            raise
    
    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall."""
        try:
            var = self._calculate_value_at_risk(returns)
            return returns[returns <= var].mean()
        except Exception as e:
            logger.error(f"Failed to calculate Expected Shortfall: {str(e)}")
            raise
    
    def _calculate_m2_measure(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate M2 measure."""
        try:
            sharpe_ratio = (returns.mean() - self.config.risk_free_rate) / returns.std()
            benchmark_sharpe = (benchmark_returns.mean() - self.config.risk_free_rate) / benchmark_returns.std()
            return sharpe_ratio * benchmark_returns.std() + self.config.risk_free_rate
        except Exception as e:
            logger.error(f"Failed to calculate M2 measure: {str(e)}")
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
        weights = pd.DataFrame(
            np.random.dirichlet(np.ones(n_assets), size=252),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.015, 252),
            index=dates
        )
        
        # Initialize performance analyzer
        config = PerformanceConfig(
            attribution_type=AttributionType.BRINSON,
            benchmark="SPY",
            risk_free_rate=0.02,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            attribution_factors=["Sector", "Style", "Region"],
            custom_weights={
                "Sector": 0.4,
                "Style": 0.3,
                "Region": 0.3
            },
            transaction_costs=0.001,
            tax_rate=0.2,
            rebalancing_frequency="M",
            attribution_threshold=0.01
        )
        
        analyzer = PerformanceAnalyzer(config)
        
        # Analyze performance
        result = analyzer.analyze_performance(returns, weights, benchmark_returns)
        
        # Print results
        print("\nPerformance Analysis Results:")
        print("\nPerformance Metrics:")
        print(result['performance_metrics'])
        
        print("\nAttribution:")
        print(result['attribution'])
        
        print("\nRisk-Adjusted Returns:")
        print(result['risk_adjusted'])
        
        print("\nTransaction Costs:")
        print(result['transaction_costs'])
        
        print("\nTax Impact:")
        print(result['tax_impact'])
        
        print("\nFactor Attribution:")
        print(result['factor_attribution'])
        
        print("\nSector Attribution:")
        print(result['sector_attribution'])
        
        print("\nStyle Attribution:")
        print(result['style_attribution'])
        
        print("\nRegion Attribution:")
        print(result['region_attribution'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 
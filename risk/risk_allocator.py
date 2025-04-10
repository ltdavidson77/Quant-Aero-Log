# ==========================
# risk/risk_allocator.py
# ==========================
# Risk allocation and budgeting.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager, RiskMetricType
from risk.portfolio_optimizer import PortfolioOptimizer, OptimizationType
from risk.drawdown_manager import DrawdownManager, DrawdownType

logger = get_logger("risk_allocator")

class AllocationType(Enum):
    """Types of risk allocation."""
    EQUAL = "equal"          # Equal risk allocation
    RISK_PARITY = "risk_parity"  # Risk parity allocation
    MIN_VAR = "min_var"      # Minimum variance allocation
    MAX_DIVERSITY = "max_diversity"  # Maximum diversity allocation
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman allocation
    CONDITIONAL = "conditional"  # Conditional risk allocation
    REGIME = "regime"       # Regime-based allocation
    DYNAMIC = "dynamic"     # Dynamic risk allocation
    CONSTRAINED = "constrained"  # Constrained risk allocation
    CUSTOM = "custom"       # Custom risk allocation

@dataclass
class RiskAllocationConfig:
    """Configuration for risk allocation."""
    allocation_type: AllocationType = AllocationType.RISK_PARITY
    risk_budget: Dict[str, float] = None  # Risk budget for each asset
    max_risk: float = 0.2  # Maximum portfolio risk
    min_risk: float = 0.05  # Minimum portfolio risk
    target_risk: float = 0.1  # Target portfolio risk
    risk_tolerance: float = 0.1  # Risk tolerance
    lookback_window: int = 252  # Lookback window for risk calculation
    min_observations: int = 20  # Minimum observations required
    confidence_level: float = 0.95  # Confidence level for risk metrics
    regime_aware: bool = True  # Whether to consider market regimes
    regime_threshold: float = 0.7  # Threshold for regime classification
    correlation_threshold: float = 0.7  # Maximum correlation threshold
    sector_limits: Dict[str, float] = None  # Sector risk limits
    asset_limits: Dict[str, float] = None  # Asset risk limits
    rebalance_threshold: float = 0.1  # Rebalancing threshold
    min_rebalance_interval: int = 5  # Minimum rebalancing interval
    transaction_costs: float = 0.001  # Transaction costs
    optimization_method: str = "SLSQP"  # Optimization method
    optimization_tolerance: float = 1e-6  # Optimization tolerance
    max_iterations: int = 1000  # Maximum optimization iterations

class RiskAllocator:
    """Manages risk allocation and budgeting."""
    
    def __init__(self, config: RiskAllocationConfig):
        self.config = config
        if self.config.risk_budget is None:
            self.config.risk_budget = {}
        if self.config.sector_limits is None:
            self.config.sector_limits = {}
        if self.config.asset_limits is None:
            self.config.asset_limits = {}
        self._setup_risk_allocator()
    
    def _setup_risk_allocator(self) -> None:
        """Initialize risk allocator."""
        try:
            logger.info("Initializing risk allocator")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup risk allocator: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate risk allocation configuration."""
        try:
            if self.config.max_risk <= self.config.min_risk:
                raise ValueError("Maximum risk must be greater than minimum risk")
            if self.config.target_risk < self.config.min_risk or self.config.target_risk > self.config.max_risk:
                raise ValueError("Target risk must be between minimum and maximum risk")
            if self.config.risk_tolerance <= 0:
                raise ValueError("Risk tolerance must be positive")
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.regime_threshold <= 0 or self.config.regime_threshold >= 1:
                raise ValueError("Regime threshold must be between 0 and 1")
            if self.config.correlation_threshold <= 0 or self.config.correlation_threshold >= 1:
                raise ValueError("Correlation threshold must be between 0 and 1")
            if self.config.rebalance_threshold <= 0:
                raise ValueError("Rebalancing threshold must be positive")
            if self.config.min_rebalance_interval < 1:
                raise ValueError("Minimum rebalancing interval must be at least 1")
            if self.config.transaction_costs < 0:
                raise ValueError("Transaction costs must be non-negative")
            if self.config.optimization_tolerance <= 0:
                raise ValueError("Optimization tolerance must be positive")
            if self.config.max_iterations < 1:
                raise ValueError("Maximum iterations must be at least 1")
        except Exception as e:
            logger.error(f"Invalid risk allocation configuration: {str(e)}")
            raise
    
    def allocate_risk(self,
                     returns: pd.DataFrame,
                     risk_metrics_manager: Optional[RiskMetricsManager] = None,
                     portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                     drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """
        Allocate risk across assets.
        
        Args:
            returns: DataFrame of asset returns
            risk_metrics_manager: Risk metrics manager (optional)
            portfolio_optimizer: Portfolio optimizer (optional)
            drawdown_manager: Drawdown manager (optional)
            
        Returns:
            Dictionary containing risk allocation results
        """
        try:
            if self.config.allocation_type == AllocationType.EQUAL:
                return self._allocate_equal_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.RISK_PARITY:
                return self._allocate_risk_parity(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.MIN_VAR:
                return self._allocate_min_var(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.MAX_DIVERSITY:
                return self._allocate_max_diversity(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.BLACK_LITTERMAN:
                return self._allocate_black_litterman(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.CONDITIONAL:
                return self._allocate_conditional_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.REGIME:
                return self._allocate_regime_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.DYNAMIC:
                return self._allocate_dynamic_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.CONSTRAINED:
                return self._allocate_constrained_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            elif self.config.allocation_type == AllocationType.CUSTOM:
                return self._allocate_custom_risk(returns, risk_metrics_manager, portfolio_optimizer, drawdown_manager)
            else:
                raise ValueError(f"Unsupported allocation type: {self.config.allocation_type}")
        except Exception as e:
            logger.error(f"Failed to allocate risk: {str(e)}")
            raise
    
    def _allocate_equal_risk(self,
                           returns: pd.DataFrame,
                           risk_metrics_manager: Optional[RiskMetricsManager] = None,
                           portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                           drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate equal risk."""
        try:
            # Calculate equal weights
            n_assets = len(returns.columns)
            weights = pd.Series(1.0 / n_assets, index=returns.columns)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate equal risk: {str(e)}")
            raise
    
    def _allocate_risk_parity(self,
                            returns: pd.DataFrame,
                            risk_metrics_manager: Optional[RiskMetricsManager] = None,
                            portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                            drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate risk parity."""
        try:
            if risk_metrics_manager is None:
                raise ValueError("Risk metrics manager is required for risk parity allocation")
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(returns, risk_metrics_manager)
            
            # Optimize weights for risk parity
            weights = self._optimize_risk_parity(risk_contributions)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate risk parity: {str(e)}")
            raise
    
    def _allocate_min_var(self,
                        returns: pd.DataFrame,
                        risk_metrics_manager: Optional[RiskMetricsManager] = None,
                        portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                        drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate minimum variance."""
        try:
            if portfolio_optimizer is None:
                raise ValueError("Portfolio optimizer is required for minimum variance allocation")
            
            # Optimize weights for minimum variance
            weights = portfolio_optimizer.optimize_portfolio(
                returns,
                OptimizationType.MIN_VAR
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate minimum variance: {str(e)}")
            raise
    
    def _allocate_max_diversity(self,
                              returns: pd.DataFrame,
                              risk_metrics_manager: Optional[RiskMetricsManager] = None,
                              portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                              drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate maximum diversity."""
        try:
            if portfolio_optimizer is None:
                raise ValueError("Portfolio optimizer is required for maximum diversity allocation")
            
            # Optimize weights for maximum diversity
            weights = portfolio_optimizer.optimize_portfolio(
                returns,
                OptimizationType.MAX_DIVERSITY
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate maximum diversity: {str(e)}")
            raise
    
    def _allocate_black_litterman(self,
                                returns: pd.DataFrame,
                                risk_metrics_manager: Optional[RiskMetricsManager] = None,
                                portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                                drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate Black-Litterman."""
        try:
            if portfolio_optimizer is None:
                raise ValueError("Portfolio optimizer is required for Black-Litterman allocation")
            
            # Optimize weights using Black-Litterman model
            weights = portfolio_optimizer.optimize_portfolio(
                returns,
                OptimizationType.BLACK_LITTERMAN
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate Black-Litterman: {str(e)}")
            raise
    
    def _allocate_conditional_risk(self,
                                 returns: pd.DataFrame,
                                 risk_metrics_manager: Optional[RiskMetricsManager] = None,
                                 portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                                 drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate conditional risk."""
        try:
            if risk_metrics_manager is None:
                raise ValueError("Risk metrics manager is required for conditional risk allocation")
            
            # Calculate conditional risk metrics
            conditional_metrics = risk_metrics_manager.calculate_risk_metrics(
                returns,
                RiskMetricType.CONDITIONAL
            )
            
            # Optimize weights based on conditional risk
            weights = self._optimize_conditional_risk(conditional_metrics)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate conditional risk: {str(e)}")
            raise
    
    def _allocate_regime_risk(self,
                            returns: pd.DataFrame,
                            risk_metrics_manager: Optional[RiskMetricsManager] = None,
                            portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                            drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate regime-based risk."""
        try:
            if drawdown_manager is None:
                raise ValueError("Drawdown manager is required for regime-based risk allocation")
            
            # Calculate regime-based drawdown
            regime_drawdown = drawdown_manager.analyze_drawdown(
                returns,
                drawdown_type=DrawdownType.REGIME
            )
            
            # Optimize weights based on regime
            weights = self._optimize_regime_risk(regime_drawdown)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate regime-based risk: {str(e)}")
            raise
    
    def _allocate_dynamic_risk(self,
                             returns: pd.DataFrame,
                             risk_metrics_manager: Optional[RiskMetricsManager] = None,
                             portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                             drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate dynamic risk."""
        try:
            # Calculate dynamic weights
            weights = self._calculate_dynamic_weights(returns)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate dynamic risk: {str(e)}")
            raise
    
    def _allocate_constrained_risk(self,
                                 returns: pd.DataFrame,
                                 risk_metrics_manager: Optional[RiskMetricsManager] = None,
                                 portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                                 drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate constrained risk."""
        try:
            # Define constraints
            constraints = self._define_constraints(returns)
            
            # Optimize weights with constraints
            weights = self._optimize_constrained_weights(returns, constraints)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate constrained risk: {str(e)}")
            raise
    
    def _allocate_custom_risk(self,
                            returns: pd.DataFrame,
                            risk_metrics_manager: Optional[RiskMetricsManager] = None,
                            portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                            drawdown_manager: Optional[DrawdownManager] = None) -> Dict[str, Any]:
        """Allocate custom risk."""
        try:
            # Apply custom risk budget
            weights = self._apply_custom_risk_budget(returns)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to allocate custom risk: {str(e)}")
            raise
    
    def _calculate_risk_contributions(self,
                                   returns: pd.DataFrame,
                                   risk_metrics_manager: RiskMetricsManager) -> pd.DataFrame:
        """Calculate risk contributions."""
        try:
            # Calculate risk metrics
            risk_metrics = risk_metrics_manager.calculate_risk_metrics(returns)
            
            # Calculate risk contributions
            risk_contributions = pd.DataFrame(index=returns.columns, columns=['contribution'])
            
            for asset in returns.columns:
                risk_contributions.loc[asset, 'contribution'] = (
                    risk_metrics['volatility'][asset] *
                    risk_metrics['correlation'][asset].mean()
                )
            
            return risk_contributions
            
        except Exception as e:
            logger.error(f"Failed to calculate risk contributions: {str(e)}")
            raise
    
    def _optimize_risk_parity(self, risk_contributions: pd.DataFrame) -> pd.Series:
        """Optimize weights for risk parity."""
        try:
            # Define objective function
            def objective(weights):
                return np.sum((weights * risk_contributions['contribution'] - 1.0 / len(weights)) ** 2)
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'ineq', 'fun': lambda x: x}
            ]
            
            # Optimize
            result = minimize(
                objective,
                np.ones(len(risk_contributions)) / len(risk_contributions),
                method=self.config.optimization_method,
                constraints=constraints,
                tol=self.config.optimization_tolerance,
                options={'maxiter': self.config.max_iterations}
            )
            
            return pd.Series(result.x, index=risk_contributions.index)
            
        except Exception as e:
            logger.error(f"Failed to optimize risk parity: {str(e)}")
            raise
    
    def _optimize_conditional_risk(self, conditional_metrics: Dict[str, Any]) -> pd.Series:
        """Optimize weights based on conditional risk."""
        try:
            # Define objective function
            def objective(weights):
                return np.sum(weights * conditional_metrics['conditional_risk'])
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'ineq', 'fun': lambda x: x}
            ]
            
            # Optimize
            result = minimize(
                objective,
                np.ones(len(conditional_metrics['conditional_risk'])) / len(conditional_metrics['conditional_risk']),
                method=self.config.optimization_method,
                constraints=constraints,
                tol=self.config.optimization_tolerance,
                options={'maxiter': self.config.max_iterations}
            )
            
            return pd.Series(result.x, index=conditional_metrics['conditional_risk'].index)
            
        except Exception as e:
            logger.error(f"Failed to optimize conditional risk: {str(e)}")
            raise
    
    def _optimize_regime_risk(self, regime_drawdown: Dict[str, Any]) -> pd.Series:
        """Optimize weights based on regime."""
        try:
            # Define objective function
            def objective(weights):
                return np.sum(weights * regime_drawdown['drawdown'].mean())
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                {'type': 'ineq', 'fun': lambda x: x}
            ]
            
            # Optimize
            result = minimize(
                objective,
                np.ones(len(regime_drawdown['drawdown'].columns)) / len(regime_drawdown['drawdown'].columns),
                method=self.config.optimization_method,
                constraints=constraints,
                tol=self.config.optimization_tolerance,
                options={'maxiter': self.config.max_iterations}
            )
            
            return pd.Series(result.x, index=regime_drawdown['drawdown'].columns)
            
        except Exception as e:
            logger.error(f"Failed to optimize regime risk: {str(e)}")
            raise
    
    def _calculate_dynamic_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate dynamic weights."""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(self.config.lookback_window).std()
            
            # Calculate inverse volatility weights
            inv_vol = 1.0 / rolling_vol.iloc[-1]
            weights = inv_vol / inv_vol.sum()
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to calculate dynamic weights: {str(e)}")
            raise
    
    def _define_constraints(self, returns: pd.DataFrame) -> List[Dict[str, Any]]:
        """Define optimization constraints."""
        try:
            constraints = []
            
            # Add weight constraints
            for asset in returns.columns:
                if asset in self.config.asset_limits:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, i=asset: x[returns.columns.get_loc(i)] - self.config.asset_limits[i]
                    })
            
            # Add sector constraints
            for sector, limit in self.config.sector_limits.items():
                sector_assets = [asset for asset in returns.columns if asset.startswith(sector)]
                if sector_assets:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, assets=sector_assets: limit - np.sum([x[returns.columns.get_loc(a)] for a in assets])
                    })
            
            return constraints
            
        except Exception as e:
            logger.error(f"Failed to define constraints: {str(e)}")
            raise
    
    def _optimize_constrained_weights(self,
                                    returns: pd.DataFrame,
                                    constraints: List[Dict[str, Any]]) -> pd.Series:
        """Optimize weights with constraints."""
        try:
            # Define objective function
            def objective(weights):
                return np.sum(weights * returns.std())
            
            # Add weight sum constraint
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            
            # Optimize
            result = minimize(
                objective,
                np.ones(len(returns.columns)) / len(returns.columns),
                method=self.config.optimization_method,
                constraints=constraints,
                tol=self.config.optimization_tolerance,
                options={'maxiter': self.config.max_iterations}
            )
            
            return pd.Series(result.x, index=returns.columns)
            
        except Exception as e:
            logger.error(f"Failed to optimize constrained weights: {str(e)}")
            raise
    
    def _apply_custom_risk_budget(self, returns: pd.DataFrame) -> pd.Series:
        """Apply custom risk budget."""
        try:
            # Initialize weights
            weights = pd.Series(0.0, index=returns.columns)
            
            # Apply risk budget
            for asset, budget in self.config.risk_budget.items():
                if asset in weights.index:
                    weights[asset] = budget
            
            # Normalize weights
            weights = weights / weights.sum()
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to apply custom risk budget: {str(e)}")
            raise
    
    def _calculate_portfolio_metrics(self,
                                   returns: pd.DataFrame,
                                   weights: pd.Series) -> Dict[str, Any]:
        """Calculate portfolio metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Calculate metrics
            metrics = {
                'return': portfolio_returns.mean(),
                'volatility': portfolio_returns.std(),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std(),
                'max_drawdown': portfolio_returns.min(),
                'correlation': returns.corr().mean().mean()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio metrics: {str(e)}")
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
        
        # Initialize risk allocator
        config = RiskAllocationConfig(
            allocation_type=AllocationType.RISK_PARITY,
            risk_budget={
                'Asset_0': 0.2,
                'Asset_1': 0.2,
                'Asset_2': 0.2,
                'Asset_3': 0.2,
                'Asset_4': 0.2
            },
            max_risk=0.2,
            min_risk=0.05,
            target_risk=0.1,
            risk_tolerance=0.1,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            regime_aware=True,
            regime_threshold=0.7,
            correlation_threshold=0.7,
            sector_limits={
                'Sector_A': 0.3,
                'Sector_B': 0.3
            },
            asset_limits={
                'Asset_0': 0.4,
                'Asset_1': 0.4
            },
            rebalance_threshold=0.1,
            min_rebalance_interval=5,
            transaction_costs=0.001,
            optimization_method="SLSQP",
            optimization_tolerance=1e-6,
            max_iterations=1000
        )
        
        allocator = RiskAllocator(config)
        
        # Allocate risk
        result = allocator.allocate_risk(returns)
        
        # Print results
        print("\nRisk Allocation Results:")
        print("\nWeights:")
        print(result['weights'])
        
        print("\nPortfolio Metrics:")
        print(result['portfolio_metrics'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 
# ==========================
# risk/portfolio_optimizer.py
# ==========================
# Portfolio optimization strategies.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
from cvxpy import *
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager, RiskMetricsConfig

logger = get_logger("portfolio_optimizer")

class OptimizationType(Enum):
    """Types of portfolio optimization strategies."""
    MEAN_VARIANCE = "mean_variance"      # Mean-Variance Optimization
    RISK_PARITY = "risk_parity"          # Risk Parity
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman Model
    HIERARCHICAL = "hierarchical"        # Hierarchical Risk Parity
    MIN_VARIANCE = "min_variance"        # Minimum Variance
    MAX_SHARPE = "max_sharpe"            # Maximum Sharpe Ratio
    MAX_DIVERSIFICATION = "max_div"      # Maximum Diversification
    EQUAL_WEIGHT = "equal_weight"        # Equal Weight
    EQUAL_RISK = "equal_risk"            # Equal Risk Contribution
    CUSTOM = "custom"                    # Custom Optimization

@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    optimization_type: OptimizationType
    risk_free_rate: float = 0.02  # Risk-free rate
    lookback_window: int = 252  # Lookback window for calculations
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    target_return: float = 0.1  # Target return for optimization
    risk_aversion: float = 1.0  # Risk aversion parameter
    transaction_cost: float = 0.001  # Transaction cost per trade
    turnover_limit: float = 0.2  # Maximum turnover allowed
    rebalance_threshold: float = 0.1  # Threshold for rebalancing
    min_observations: int = 20  # Minimum observations required
    confidence_interval: float = 0.95  # Confidence interval for estimates
    black_litterman_views: List[Dict[str, float]] = None  # Black-Litterman views
    hierarchical_clusters: int = 5  # Number of clusters for hierarchical optimization
    custom_objective: str = None  # Custom objective function
    custom_constraints: List[str] = None  # Custom constraints

class PortfolioOptimizer:
    """Manages portfolio optimization strategies."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self._setup_portfolio_optimizer()
    
    def _setup_portfolio_optimizer(self) -> None:
        """Initialize portfolio optimizer."""
        try:
            logger.info("Initializing portfolio optimizer")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup portfolio optimizer: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate portfolio configuration."""
        try:
            if self.config.risk_free_rate < 0:
                raise ValueError("Risk-free rate must be non-negative")
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.min_weight < 0 or self.config.min_weight > 1:
                raise ValueError("Minimum weight must be between 0 and 1")
            if self.config.max_weight < 0 or self.config.max_weight > 1:
                raise ValueError("Maximum weight must be between 0 and 1")
            if self.config.min_weight > self.config.max_weight:
                raise ValueError("Minimum weight must be less than maximum weight")
            if self.config.transaction_cost < 0:
                raise ValueError("Transaction cost must be non-negative")
            if self.config.turnover_limit <= 0 or self.config.turnover_limit > 1:
                raise ValueError("Turnover limit must be between 0 and 1")
            if self.config.rebalance_threshold <= 0 or self.config.rebalance_threshold > 1:
                raise ValueError("Rebalance threshold must be between 0 and 1")
            if self.config.confidence_interval <= 0 or self.config.confidence_interval >= 1:
                raise ValueError("Confidence interval must be between 0 and 1")
            if self.config.hierarchical_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
        except Exception as e:
            logger.error(f"Invalid portfolio configuration: {str(e)}")
            raise
    
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         current_weights: Optional[np.ndarray] = None,
                         benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: DataFrame of asset returns
            current_weights: Current portfolio weights (optional)
            benchmark_weights: Benchmark portfolio weights (optional)
            
        Returns:
            Dictionary containing optimized weights and metrics
        """
        try:
            if self.config.optimization_type == OptimizationType.MEAN_VARIANCE:
                return self._optimize_mean_variance(returns)
            elif self.config.optimization_type == OptimizationType.RISK_PARITY:
                return self._optimize_risk_parity(returns)
            elif self.config.optimization_type == OptimizationType.BLACK_LITTERMAN:
                return self._optimize_black_litterman(returns, benchmark_weights)
            elif self.config.optimization_type == OptimizationType.HIERARCHICAL:
                return self._optimize_hierarchical(returns)
            elif self.config.optimization_type == OptimizationType.MIN_VARIANCE:
                return self._optimize_min_variance(returns)
            elif self.config.optimization_type == OptimizationType.MAX_SHARPE:
                return self._optimize_max_sharpe(returns)
            elif self.config.optimization_type == OptimizationType.MAX_DIVERSIFICATION:
                return self._optimize_max_diversification(returns)
            elif self.config.optimization_type == OptimizationType.EQUAL_WEIGHT:
                return self._optimize_equal_weight(returns)
            elif self.config.optimization_type == OptimizationType.EQUAL_RISK:
                return self._optimize_equal_risk(returns)
            elif self.config.optimization_type == OptimizationType.CUSTOM:
                return self._optimize_custom(returns)
            else:
                raise ValueError(f"Unsupported optimization type: {self.config.optimization_type}")
        except Exception as e:
            logger.error(f"Failed to optimize portfolio: {str(e)}")
            raise
    
    def _optimize_mean_variance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using Mean-Variance optimization."""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define objective
            portfolio_return = expected_returns @ weights
            portfolio_risk = quad_form(weights, cov_matrix)
            objective = Maximize(portfolio_return - self.config.risk_aversion * portfolio_risk)
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize mean-variance portfolio: {str(e)}")
            raise
    
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using Risk Parity."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Calculate risk contributions
            portfolio_risk = sqrt(quad_form(weights, cov_matrix))
            risk_contributions = []
            for i in range(n_assets):
                risk_contributions.append(
                    (weights[i] * (cov_matrix @ weights)[i]) / portfolio_risk
                )
            
            # Define objective
            objective = Minimize(sum_squares(risk_contributions))
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize risk parity portfolio: {str(e)}")
            raise
    
    def _optimize_black_litterman(self,
                                returns: pd.DataFrame,
                                benchmark_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize portfolio using Black-Litterman model."""
        try:
            # Calculate market implied returns
            cov_matrix = returns.cov()
            if benchmark_weights is None:
                benchmark_weights = np.ones(len(returns.columns)) / len(returns.columns)
            
            market_returns = self.config.risk_aversion * cov_matrix @ benchmark_weights
            
            # Incorporate views
            if self.config.black_litterman_views:
                P = np.zeros((len(self.config.black_litterman_views), len(returns.columns)))
                Q = np.zeros(len(self.config.black_litterman_views))
                
                for i, view in enumerate(self.config.black_litterman_views):
                    P[i, view['asset']] = view['weight']
                    Q[i] = view['return']
                
                # Calculate posterior returns
                tau = 0.05
                omega = np.diag(np.diag(P @ cov_matrix @ P.T))
                posterior_returns = np.linalg.inv(
                    np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P
                ) @ (
                    np.linalg.inv(tau * cov_matrix) @ market_returns + P.T @ np.linalg.inv(omega) @ Q
                )
            else:
                posterior_returns = market_returns
            
            # Optimize using mean-variance with posterior returns
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define objective
            portfolio_return = posterior_returns @ weights
            portfolio_risk = quad_form(weights, cov_matrix)
            objective = Maximize(portfolio_return - self.config.risk_aversion * portfolio_risk)
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize Black-Litterman portfolio: {str(e)}")
            raise
    
    def _optimize_hierarchical(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using Hierarchical Risk Parity."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            Z = linkage(corr_matrix, method='ward')
            clusters = fcluster(Z, self.config.hierarchical_clusters, criterion='maxclust')
            
            # Initialize weights
            n_assets = len(returns.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Optimize within clusters
            for cluster in range(1, self.config.hierarchical_clusters + 1):
                cluster_assets = np.where(clusters == cluster)[0]
                if len(cluster_assets) > 1:
                    cluster_returns = returns.iloc[:, cluster_assets]
                    cluster_weights = self._optimize_risk_parity(cluster_returns)['weights']
                    weights[cluster_assets] = cluster_weights
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Apply constraints
            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
            weights = weights / weights.sum()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize hierarchical portfolio: {str(e)}")
            raise
    
    def _optimize_min_variance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for minimum variance."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define objective
            portfolio_risk = quad_form(weights, cov_matrix)
            objective = Minimize(portfolio_risk)
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize minimum variance portfolio: {str(e)}")
            raise
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for maximum Sharpe ratio."""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define objective
            portfolio_return = expected_returns @ weights
            portfolio_risk = quad_form(weights, cov_matrix)
            sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / sqrt(portfolio_risk)
            objective = Maximize(sharpe_ratio)
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize maximum Sharpe portfolio: {str(e)}")
            raise
    
    def _optimize_max_diversification(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for maximum diversification."""
        try:
            # Calculate covariance matrix and volatilities
            cov_matrix = returns.cov()
            volatilities = np.sqrt(np.diag(cov_matrix))
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define objective
            portfolio_risk = sqrt(quad_form(weights, cov_matrix))
            weighted_vol = weights @ volatilities
            diversification_ratio = weighted_vol / portfolio_risk
            objective = Maximize(diversification_ratio)
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize maximum diversification portfolio: {str(e)}")
            raise
    
    def _optimize_equal_weight(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using equal weights."""
        try:
            # Calculate equal weights
            n_assets = len(returns.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return {
                'weights': weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize equal weight portfolio: {str(e)}")
            raise
    
    def _optimize_equal_risk(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio for equal risk contribution."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Calculate risk contributions
            portfolio_risk = sqrt(quad_form(weights, cov_matrix))
            risk_contributions = []
            for i in range(n_assets):
                risk_contributions.append(
                    (weights[i] * (cov_matrix @ weights)[i]) / portfolio_risk
                )
            
            # Define objective
            target_contribution = 1 / n_assets
            objective = Minimize(sum_squares([rc - target_contribution for rc in risk_contributions]))
            
            # Define constraints
            constraints = [
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ]
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize equal risk portfolio: {str(e)}")
            raise
    
    def _optimize_custom(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using custom objective and constraints."""
        try:
            if not self.config.custom_objective or not self.config.custom_constraints:
                raise ValueError("Custom objective and constraints must be provided")
            
            # Define optimization problem
            n_assets = len(returns.columns)
            weights = Variable(n_assets)
            
            # Define custom objective
            objective = eval(self.config.custom_objective)
            
            # Define custom constraints
            constraints = [eval(constraint) for constraint in self.config.custom_constraints]
            
            # Add basic constraints
            constraints.extend([
                sum(weights) == 1,
                weights >= self.config.min_weight,
                weights <= self.config.max_weight
            ])
            
            # Solve optimization problem
            prob = Problem(objective, constraints)
            prob.solve()
            
            # Get optimized weights
            optimized_weights = np.array(weights.value).flatten()
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(returns, optimized_weights)
            
            return {
                'weights': optimized_weights,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize custom portfolio: {str(e)}")
            raise
    
    def _calculate_portfolio_metrics(self,
                                   returns: pd.DataFrame,
                                   weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = returns @ weights
            
            # Initialize risk metrics manager
            risk_config = RiskMetricsConfig(
                var_confidence=0.95,
                cvar_confidence=0.99,
                risk_free_rate=self.config.risk_free_rate,
                lookback_window=self.config.lookback_window
            )
            risk_manager = RiskMetricsManager(risk_config)
            
            # Calculate risk metrics
            metrics = risk_manager.calculate_risk_metrics(
                portfolio_returns,
                metric_types=[
                    RiskMetricType.VAR,
                    RiskMetricType.CVAR,
                    RiskMetricType.SHARPE_RATIO,
                    RiskMetricType.SORTINO_RATIO,
                    RiskMetricType.MAX_DRAWDOWN,
                    RiskMetricType.VOLATILITY
                ]
            )
            
            # Add additional metrics
            metrics['expected_return'] = portfolio_returns.mean()
            metrics['turnover'] = np.sum(np.abs(weights - self.current_weights)) if hasattr(self, 'current_weights') else 0
            metrics['diversification_ratio'] = np.sum(weights * np.sqrt(np.diag(returns.cov()))) / np.sqrt(weights @ returns.cov() @ weights)
            
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
        
        # Initialize portfolio optimizer
        config = PortfolioConfig(
            optimization_type=OptimizationType.MEAN_VARIANCE,
            risk_free_rate=0.02,
            lookback_window=252,
            min_weight=0.0,
            max_weight=0.3,
            target_return=0.1,
            risk_aversion=1.0,
            transaction_cost=0.001,
            turnover_limit=0.2,
            rebalance_threshold=0.1,
            min_observations=20,
            confidence_interval=0.95
        )
        
        optimizer = PortfolioOptimizer(config)
        
        # Optimize portfolio
        result = optimizer.optimize_portfolio(returns)
        
        # Print results
        print("\nPortfolio Optimization Results:")
        print("\nOptimized Weights:")
        for asset, weight in zip(returns.columns, result['weights']):
            print(f"  {asset}: {weight:.4f}")
        
        print("\nPortfolio Metrics:")
        for metric, value in result['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 
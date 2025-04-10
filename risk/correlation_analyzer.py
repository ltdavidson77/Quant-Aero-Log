# ==========================
# risk/correlation_analyzer.py
# ==========================
# Correlation analysis and management.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from risk.market_regime import MarketRegimeManager, RegimeType

logger = get_logger("correlation_analyzer")

class CorrelationType(Enum):
    """Types of correlation analysis."""
    PEARSON = "pearson"      # Pearson correlation
    SPEARMAN = "spearman"    # Spearman rank correlation
    KENDALL = "kendall"      # Kendall's tau
    ROLLING = "rolling"      # Rolling correlation
    DYNAMIC = "dynamic"      # Dynamic correlation
    CONDITIONAL = "conditional"  # Conditional correlation
    PARTIAL = "partial"      # Partial correlation
    DISTANCE = "distance"    # Distance correlation
    COINTEGRATION = "cointegration"  # Cointegration analysis
    CAUSALITY = "causality"  # Granger causality

@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    correlation_type: CorrelationType = CorrelationType.PEARSON
    lookback_window: int = 252  # Lookback window for correlation calculation
    min_observations: int = 20  # Minimum observations required
    confidence_level: float = 0.95  # Confidence level for significance testing
    rolling_window: int = 60  # Window for rolling correlation
    regime_aware: bool = True  # Whether to consider market regimes
    regime_threshold: float = 0.7  # Threshold for regime classification
    clustering_method: str = "hierarchical"  # Clustering method: "hierarchical", "kmeans"
    n_clusters: int = 5  # Number of clusters for clustering
    distance_metric: str = "euclidean"  # Distance metric for clustering
    cointegration_order: int = 1  # Order for cointegration test
    causality_lags: int = 5  # Number of lags for causality test
    significance_threshold: float = 0.05  # Significance threshold for tests

class CorrelationAnalyzer:
    """Manages correlation analysis and management."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self._setup_correlation_analyzer()
    
    def _setup_correlation_analyzer(self) -> None:
        """Initialize correlation analyzer."""
        try:
            logger.info("Initializing correlation analyzer")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup correlation analyzer: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate correlation configuration."""
        try:
            if self.config.lookback_window < self.config.min_observations:
                raise ValueError("Lookback window must be at least min_observations")
            if self.config.confidence_level <= 0 or self.config.confidence_level >= 1:
                raise ValueError("Confidence level must be between 0 and 1")
            if self.config.rolling_window < self.config.min_observations:
                raise ValueError("Rolling window must be at least min_observations")
            if self.config.regime_threshold <= 0 or self.config.regime_threshold >= 1:
                raise ValueError("Regime threshold must be between 0 and 1")
            if self.config.n_clusters < 2:
                raise ValueError("Number of clusters must be at least 2")
            if self.config.cointegration_order < 1:
                raise ValueError("Cointegration order must be at least 1")
            if self.config.causality_lags < 1:
                raise ValueError("Number of causality lags must be at least 1")
            if self.config.significance_threshold <= 0 or self.config.significance_threshold >= 1:
                raise ValueError("Significance threshold must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid correlation configuration: {str(e)}")
            raise
    
    def analyze_correlation(self,
                          returns: pd.DataFrame,
                          regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """
        Analyze correlations between assets.
        
        Args:
            returns: DataFrame of asset returns
            regime_manager: Market regime manager (optional)
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            if self.config.correlation_type == CorrelationType.PEARSON:
                return self._analyze_pearson(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.SPEARMAN:
                return self._analyze_spearman(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.KENDALL:
                return self._analyze_kendall(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.ROLLING:
                return self._analyze_rolling(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.DYNAMIC:
                return self._analyze_dynamic(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.CONDITIONAL:
                return self._analyze_conditional(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.PARTIAL:
                return self._analyze_partial(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.DISTANCE:
                return self._analyze_distance(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.COINTEGRATION:
                return self._analyze_cointegration(returns, regime_manager)
            elif self.config.correlation_type == CorrelationType.CAUSALITY:
                return self._analyze_causality(returns, regime_manager)
            else:
                raise ValueError(f"Unsupported correlation type: {self.config.correlation_type}")
        except Exception as e:
            logger.error(f"Failed to analyze correlations: {str(e)}")
            raise
    
    def _analyze_pearson(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze Pearson correlations."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Calculate significance
            n = len(returns)
            t_stat = corr_matrix * np.sqrt((n - 2) / (1 - corr_matrix**2))
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            
            # Get significant correlations
            significant = p_values < self.config.significance_threshold
            
            # Cluster assets if needed
            clusters = self._cluster_assets(corr_matrix) if self.config.clustering_method else None
            
            return {
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'significant_correlations': significant,
                'clusters': clusters
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze Pearson correlations: {str(e)}")
            raise
    
    def _analyze_spearman(self,
                         returns: pd.DataFrame,
                         regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze Spearman rank correlations."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr(method='spearman')
            
            # Calculate significance
            n = len(returns)
            t_stat = corr_matrix * np.sqrt((n - 2) / (1 - corr_matrix**2))
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            
            # Get significant correlations
            significant = p_values < self.config.significance_threshold
            
            # Cluster assets if needed
            clusters = self._cluster_assets(corr_matrix) if self.config.clustering_method else None
            
            return {
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'significant_correlations': significant,
                'clusters': clusters
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze Spearman correlations: {str(e)}")
            raise
    
    def _analyze_kendall(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze Kendall's tau correlations."""
        try:
            # Calculate correlation matrix
            corr_matrix = returns.corr(method='kendall')
            
            # Calculate significance
            n = len(returns)
            z_stat = corr_matrix * np.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
            
            # Get significant correlations
            significant = p_values < self.config.significance_threshold
            
            # Cluster assets if needed
            clusters = self._cluster_assets(corr_matrix) if self.config.clustering_method else None
            
            return {
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'significant_correlations': significant,
                'clusters': clusters
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze Kendall correlations: {str(e)}")
            raise
    
    def _analyze_rolling(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze rolling correlations."""
        try:
            # Initialize results
            rolling_correlations = {}
            rolling_p_values = {}
            
            # Calculate rolling correlations
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Calculate rolling correlation
                    rolling_corr = returns[asset1].rolling(
                        self.config.rolling_window
                    ).corr(returns[asset2])
                    
                    # Calculate rolling p-values
                    n = self.config.rolling_window
                    t_stat = rolling_corr * np.sqrt((n - 2) / (1 - rolling_corr**2))
                    rolling_p = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
                    
                    rolling_correlations[f"{asset1}_{asset2}"] = rolling_corr
                    rolling_p_values[f"{asset1}_{asset2}"] = rolling_p
            
            return {
                'rolling_correlations': rolling_correlations,
                'rolling_p_values': rolling_p_values
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze rolling correlations: {str(e)}")
            raise
    
    def _analyze_dynamic(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze dynamic correlations."""
        try:
            # Initialize results
            dynamic_correlations = {}
            
            # Calculate dynamic correlations
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Calculate dynamic correlation
                    dynamic_corr = self._calculate_dynamic_correlation(
                        returns[asset1],
                        returns[asset2],
                        regime_manager
                    )
                    
                    dynamic_correlations[f"{asset1}_{asset2}"] = dynamic_corr
            
            return {
                'dynamic_correlations': dynamic_correlations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze dynamic correlations: {str(e)}")
            raise
    
    def _analyze_conditional(self,
                           returns: pd.DataFrame,
                           regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze conditional correlations."""
        try:
            if regime_manager is None:
                raise ValueError("Regime manager is required for conditional correlation analysis")
            
            # Initialize results
            conditional_correlations = {}
            
            # Detect current regime
            regime_result = regime_manager.detect_regime(returns.mean(axis=1))
            current_regime = regime_result['regime']
            
            # Calculate conditional correlations
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Filter returns by regime
                    regime_returns = returns[
                        returns.index.isin(
                            returns.index[regime_result['regime'] == current_regime]
                        )
                    ]
                    
                    # Calculate correlation
                    corr = regime_returns[asset1].corr(regime_returns[asset2])
                    
                    conditional_correlations[f"{asset1}_{asset2}"] = corr
            
            return {
                'regime': current_regime,
                'conditional_correlations': conditional_correlations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze conditional correlations: {str(e)}")
            raise
    
    def _analyze_partial(self,
                        returns: pd.DataFrame,
                        regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze partial correlations."""
        try:
            # Initialize results
            partial_correlations = {}
            
            # Calculate partial correlations
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Get other assets
                    other_assets = [col for col in returns.columns if col not in [asset1, asset2]]
                    
                    # Calculate partial correlation
                    if len(other_assets) > 0:
                        partial_corr = self._calculate_partial_correlation(
                            returns[asset1],
                            returns[asset2],
                            returns[other_assets]
                        )
                    else:
                        partial_corr = returns[asset1].corr(returns[asset2])
                    
                    partial_correlations[f"{asset1}_{asset2}"] = partial_corr
            
            return {
                'partial_correlations': partial_correlations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze partial correlations: {str(e)}")
            raise
    
    def _analyze_distance(self,
                         returns: pd.DataFrame,
                         regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze distance correlations."""
        try:
            # Initialize results
            distance_correlations = {}
            
            # Calculate distance correlations
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Calculate distance correlation
                    dist_corr = self._calculate_distance_correlation(
                        returns[asset1],
                        returns[asset2]
                    )
                    
                    distance_correlations[f"{asset1}_{asset2}"] = dist_corr
            
            return {
                'distance_correlations': distance_correlations
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze distance correlations: {str(e)}")
            raise
    
    def _analyze_cointegration(self,
                             returns: pd.DataFrame,
                             regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze cointegration relationships."""
        try:
            # Initialize results
            cointegration_results = {}
            
            # Calculate cointegration
            for i in range(len(returns.columns)):
                for j in range(i + 1, len(returns.columns)):
                    asset1 = returns.columns[i]
                    asset2 = returns.columns[j]
                    
                    # Perform cointegration test
                    result = self._test_cointegration(
                        returns[asset1],
                        returns[asset2]
                    )
                    
                    cointegration_results[f"{asset1}_{asset2}"] = result
            
            return {
                'cointegration_results': cointegration_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze cointegration: {str(e)}")
            raise
    
    def _analyze_causality(self,
                         returns: pd.DataFrame,
                         regime_manager: Optional[MarketRegimeManager] = None) -> Dict[str, Any]:
        """Analyze Granger causality."""
        try:
            # Initialize results
            causality_results = {}
            
            # Calculate causality
            for i in range(len(returns.columns)):
                for j in range(len(returns.columns)):
                    if i != j:
                        asset1 = returns.columns[i]
                        asset2 = returns.columns[j]
                        
                        # Perform Granger causality test
                        result = self._test_granger_causality(
                            returns[asset1],
                            returns[asset2]
                        )
                        
                        causality_results[f"{asset1}_{asset2}"] = result
            
            return {
                'causality_results': causality_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze causality: {str(e)}")
            raise
    
    def _cluster_assets(self, corr_matrix: pd.DataFrame) -> Dict[str, int]:
        """Cluster assets based on correlation matrix."""
        try:
            # Convert correlation to distance
            distance_matrix = np.sqrt(2 * (1 - corr_matrix))
            
            # Perform clustering
            if self.config.clustering_method == "hierarchical":
                clustering = AgglomerativeClustering(
                    n_clusters=self.config.n_clusters,
                    affinity='precomputed',
                    linkage='complete'
                )
                clusters = clustering.fit_predict(distance_matrix)
            else:
                raise ValueError(f"Unsupported clustering method: {self.config.clustering_method}")
            
            # Map assets to clusters
            asset_clusters = {
                asset: cluster
                for asset, cluster in zip(corr_matrix.columns, clusters)
            }
            
            return asset_clusters
            
        except Exception as e:
            logger.error(f"Failed to cluster assets: {str(e)}")
            raise
    
    def _calculate_dynamic_correlation(self,
                                    returns1: pd.Series,
                                    returns2: pd.Series,
                                    regime_manager: Optional[MarketRegimeManager] = None) -> pd.Series:
        """Calculate dynamic correlation."""
        try:
            # Initialize result
            dynamic_corr = pd.Series(index=returns1.index)
            
            # Calculate rolling correlation
            for i in range(self.config.lookback_window, len(returns1)):
                window_returns1 = returns1.iloc[i-self.config.lookback_window:i]
                window_returns2 = returns2.iloc[i-self.config.lookback_window:i]
                
                # Calculate correlation
                corr = window_returns1.corr(window_returns2)
                dynamic_corr.iloc[i] = corr
            
            return dynamic_corr
            
        except Exception as e:
            logger.error(f"Failed to calculate dynamic correlation: {str(e)}")
            raise
    
    def _calculate_partial_correlation(self,
                                     returns1: pd.Series,
                                     returns2: pd.Series,
                                     other_returns: pd.DataFrame) -> float:
        """Calculate partial correlation."""
        try:
            # Create design matrix
            X = other_returns.copy()
            X['const'] = 1
            
            # Fit linear models
            model1 = stats.linregress(X, returns1)
            model2 = stats.linregress(X, returns2)
            
            # Get residuals
            residuals1 = returns1 - model1.predict(X)
            residuals2 = returns2 - model2.predict(X)
            
            # Calculate correlation between residuals
            partial_corr = residuals1.corr(residuals2)
            
            return partial_corr
            
        except Exception as e:
            logger.error(f"Failed to calculate partial correlation: {str(e)}")
            raise
    
    def _calculate_distance_correlation(self,
                                      returns1: pd.Series,
                                      returns2: pd.Series) -> float:
        """Calculate distance correlation."""
        try:
            # Calculate distance matrices
            n = len(returns1)
            A = np.zeros((n, n))
            B = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    A[i, j] = abs(returns1.iloc[i] - returns1.iloc[j])
                    B[i, j] = abs(returns2.iloc[i] - returns2.iloc[j])
            
            # Calculate distance covariance
            A_centered = A - A.mean(axis=0) - A.mean(axis=1)[:, np.newaxis] + A.mean()
            B_centered = B - B.mean(axis=0) - B.mean(axis=1)[:, np.newaxis] + B.mean()
            dcov = np.sqrt((A_centered * B_centered).sum() / (n * n))
            
            # Calculate distance variances
            dvar1 = np.sqrt((A_centered * A_centered).sum() / (n * n))
            dvar2 = np.sqrt((B_centered * B_centered).sum() / (n * n))
            
            # Calculate distance correlation
            dcorr = dcov / np.sqrt(dvar1 * dvar2)
            
            return dcorr
            
        except Exception as e:
            logger.error(f"Failed to calculate distance correlation: {str(e)}")
            raise
    
    def _test_cointegration(self,
                          returns1: pd.Series,
                          returns2: pd.Series) -> Dict[str, Any]:
        """Test for cointegration."""
        try:
            # Perform Engle-Granger test
            result = stats.coint(returns1, returns2)
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[2],
                'is_cointegrated': result[1] < self.config.significance_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to test cointegration: {str(e)}")
            raise
    
    def _test_granger_causality(self,
                              returns1: pd.Series,
                              returns2: pd.Series) -> Dict[str, Any]:
        """Test for Granger causality."""
        try:
            # Create lagged variables
            data = pd.DataFrame({
                'y': returns1,
                'x': returns2
            })
            
            for i in range(1, self.config.causality_lags + 1):
                data[f'x_lag{i}'] = data['x'].shift(i)
                data[f'y_lag{i}'] = data['y'].shift(i)
            
            # Drop NaN values
            data = data.dropna()
            
            # Fit restricted model
            restricted_formula = 'y ~ ' + ' + '.join([f'y_lag{i}' for i in range(1, self.config.causality_lags + 1)])
            restricted_model = stats.ols(restricted_formula, data).fit()
            
            # Fit unrestricted model
            unrestricted_formula = restricted_formula + ' + ' + ' + '.join([f'x_lag{i}' for i in range(1, self.config.causality_lags + 1)])
            unrestricted_model = stats.ols(unrestricted_formula, data).fit()
            
            # Calculate F-statistic
            ssr_restricted = restricted_model.ssr
            ssr_unrestricted = unrestricted_model.ssr
            n = len(data)
            k = self.config.causality_lags
            
            f_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (n - 2 * k - 1))
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k - 1)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'is_causal': p_value < self.config.significance_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to test Granger causality: {str(e)}")
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
        
        # Initialize correlation analyzer
        config = CorrelationConfig(
            correlation_type=CorrelationType.PEARSON,
            lookback_window=252,
            min_observations=20,
            confidence_level=0.95,
            rolling_window=60,
            regime_aware=True,
            regime_threshold=0.7,
            clustering_method="hierarchical",
            n_clusters=3,
            distance_metric="euclidean",
            cointegration_order=1,
            causality_lags=5,
            significance_threshold=0.05
        )
        
        analyzer = CorrelationAnalyzer(config)
        
        # Analyze correlations
        result = analyzer.analyze_correlation(returns)
        
        # Print results
        print("\nCorrelation Analysis Results:")
        print("\nCorrelation Matrix:")
        print(result['correlation_matrix'])
        
        print("\nSignificant Correlations:")
        print(result['significant_correlations'])
        
        if 'clusters' in result:
            print("\nAsset Clusters:")
            for asset, cluster in result['clusters'].items():
                print(f"  {asset}: Cluster {cluster}")
        
    except Exception as e:
        print(f"Error: {str(e)}") 
import cupy as cp
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from algorithm_support import AlgorithmBase
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import zscore
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from enum import Enum

class ClusteringMethod(Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"

class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"

@dataclass
class ClusterStats:
    size: int
    volatility: float
    momentum: float
    liquidity: float
    centrality: float
    silhouette: float

class Algorithm(AlgorithmBase):
    """Enhanced K-Means clustering algorithm with advanced features and mesh topology integration."""
    
    def get_default_config(self) -> Dict:
        """Get default configuration for the algorithm."""
        return {
            "n_clusters": 5,
            "metrics": ["Adj Close", "Vol_1d"],
            "use_advanced_metrics": True,
            "pca_components": 3,
            "volatility_window": 20,
            "momentum_window": 10,
            "clustering_method": ClusteringMethod.KMEANS,
            "distance_metric": DistanceMetric.EUCLIDEAN,
            "centrality_weight": 0.3,
            "volatility_weight": 0.2,
            "momentum_weight": 0.2,
            "liquidity_weight": 0.1,
            "max_workers": 4,
            "batch_size": 50,
            "use_gpu": True,
            "normalization_method": "standard",  # standard, robust, zscore
            "outlier_removal": True,
            "outlier_threshold": 3.0,
            "min_cluster_size": 5,
            "max_cluster_size": 50,
            "silhouette_threshold": 0.5,
            "market_regime": "normal",  # normal, volatile, trending
            "time_horizon": "short",    # short, medium, long
            "sector_weighting": True,
            "liquidity_threshold": 0.1
        }

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return ["Adj Close", "Volume", "Vol_1d", "High", "Low", "Open", "Close"]

    def get_version(self) -> str:
        """Get algorithm version."""
        return "v3.0"

    def _compute_advanced_metrics(self, data: Dict[str, Dict[str, float]], window: int) -> Dict[str, Dict[str, float]]:
        """Compute advanced financial metrics."""
        metrics = {}
        for stock, values in data.items():
            prices = np.array([values.get("Adj Close", 0.0) for _ in range(window)])
            volumes = np.array([values.get("Volume", 0.0) for _ in range(window)])
            
            # Compute returns
            returns = np.diff(prices) / prices[:-1]
            
            # Compute volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Compute momentum
            momentum = (prices[-1] / prices[0] - 1) * 100
            
            # Compute liquidity
            liquidity = np.mean(volumes) / np.mean(prices)
            
            # Compute risk-adjusted returns
            risk_adjusted_returns = np.mean(returns) / volatility if volatility > 0 else 0
            
            # Compute market regime indicators
            trend_strength = abs(np.mean(returns)) / np.std(returns) if np.std(returns) > 0 else 0
            
            metrics[stock] = {
                "volatility": volatility,
                "momentum": momentum,
                "liquidity": liquidity,
                "returns": returns[-1] if len(returns) > 0 else 0.0,
                "risk_adjusted_returns": risk_adjusted_returns,
                "trend_strength": trend_strength
            }
        return metrics

    def _prepare_features(self, data: Dict[str, Dict[str, float]], 
                         advanced_metrics: Dict[str, Dict[str, float]],
                         centrality: Dict[str, float]) -> cp.ndarray:
        """Prepare feature matrix for clustering."""
        stocks = list(data.keys())
        config = self.get_default_config()
        
        # Combine basic and advanced metrics
        features = []
        for stock in stocks:
            stock_features = []
            
            # Add basic metrics
            for metric in config["metrics"]:
                value = data[stock].get(metric, 0.0)
                stock_features.append(cp.sin(cp.log1p(value)))
            
            # Add advanced metrics
            if config["use_advanced_metrics"]:
                stock_features.extend([
                    advanced_metrics[stock]["volatility"],
                    advanced_metrics[stock]["momentum"],
                    advanced_metrics[stock]["liquidity"]
                ])
            
            # Add mesh centrality
            stock_features.append(centrality.get(stock, 0.0))
            
            features.append(stock_features)
        
        X = cp.array(features)
        
        # Apply normalization
        if config["normalization_method"] == "standard":
            X = (X - cp.mean(X, axis=0)) / cp.std(X, axis=0)
        elif config["normalization_method"] == "robust":
            X = (X - cp.median(X, axis=0)) / cp.percentile(X, 75, axis=0) - cp.percentile(X, 25, axis=0)
        else:  # zscore
            X = cp.array([zscore(x.get()) for x in X])
        
        return X

    def _remove_outliers(self, X: cp.ndarray, threshold: float) -> Tuple[cp.ndarray, List[int]]:
        """Remove outliers from the feature matrix."""
        distances = cp.linalg.norm(X - cp.mean(X, axis=0), axis=1)
        mean_dist = cp.mean(distances)
        std_dist = cp.std(distances)
        
        mask = distances <= mean_dist + threshold * std_dist
        return X[mask], cp.where(mask)[0].tolist()

    def _compute_cluster_stats(self, cluster: List[str], data: Dict[str, Dict[str, float]],
                             advanced_metrics: Dict[str, Dict[str, float]],
                             centrality: Dict[str, float]) -> ClusterStats:
        """Compute statistics for a cluster."""
        volatilities = [advanced_metrics[s]["volatility"] for s in cluster]
        momentums = [advanced_metrics[s]["momentum"] for s in cluster]
        liquidities = [advanced_metrics[s]["liquidity"] for s in cluster]
        centralities = [centrality.get(s, 0.0) for s in cluster]
        
        return ClusterStats(
            size=len(cluster),
            volatility=np.mean(volatilities),
            momentum=np.mean(momentums),
            liquidity=np.mean(liquidities),
            centrality=np.mean(centralities),
            silhouette=0.0  # Will be computed later
        )

    def run(self, api: 'AnalysisAPI', data: Dict[str, Dict[str, float]], 
            interval: str, config: Dict, support_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run enhanced K-Means clustering algorithm.
        
        Args:
            api: Analysis API instance
            data: Stock data dictionary
            interval: Time interval
            config: Algorithm configuration
            support_results: Additional support results
            
        Returns:
            Dictionary containing cluster assignments and statistics
        """
        stocks = list(data.keys())
        if len(stocks) < config["n_clusters"]:
            return {"kmeans_clusters": {}, "kmeans_centroids": [], "cluster_stats": {}}

        try:
            start_time = time.time()
            
            # Get configuration
            config = {**self.get_default_config(), **config}
            
            # Get mesh topology results
            mesh_result = api.get(f"mesh_results.{interval}", {})
            centrality = mesh_result.get("centrality", {})
            
            # Compute advanced metrics
            advanced_metrics = self._compute_advanced_metrics(data, config["volatility_window"])
            
            # Prepare feature matrix
            X = self._prepare_features(data, advanced_metrics, centrality)
            
            # Remove outliers if enabled
            if config["outlier_removal"]:
                X, valid_indices = self._remove_outliers(X, config["outlier_threshold"])
                stocks = [stocks[i] for i in valid_indices]
            
            # Apply PCA if enabled
            if config["pca_components"] > 0:
                pca = PCA(n_components=config["pca_components"])
                X = cp.array(pca.fit_transform(X.get()))
            
            # Perform clustering
            if config["clustering_method"] == ClusteringMethod.KMEANS:
                clusterer = KMeans(
                    n_clusters=config["n_clusters"],
                    random_state=42
                )
            elif config["clustering_method"] == ClusteringMethod.DBSCAN:
                clusterer = DBSCAN(
                    eps=0.5,
                    min_samples=config["min_cluster_size"]
                )
            else:  # AGGLOMERATIVE
                clusterer = AgglomerativeClustering(
                    n_clusters=config["n_clusters"],
                    linkage="ward"
                )
            
            labels = clusterer.fit_predict(X.get())
            
            # Organize clusters
            clusters = {i: [] for i in range(max(labels) + 1)}
            for stock, label in zip(stocks, labels):
                if label != -1:  # Skip noise points in DBSCAN
                    clusters[label].append(stock)
            
            # Filter clusters by size
            clusters = {k: v for k, v in clusters.items() 
                       if config["min_cluster_size"] <= len(v) <= config["max_cluster_size"]}
            
            # Compute cluster statistics
            cluster_stats = {}
            for label, cluster in clusters.items():
                stats = self._compute_cluster_stats(cluster, data, advanced_metrics, centrality)
                cluster_stats[label] = stats
            
            # Compute silhouette score
            if len(clusters) > 1:
                silhouette = silhouette_score(X.get(), labels)
            else:
                silhouette = 0.0
            
            result = {
                "kmeans_clusters": clusters,
                "kmeans_centroids": clusterer.cluster_centers_.tolist() if hasattr(clusterer, 'cluster_centers_') else [],
                "cluster_stats": {k: vars(v) for k, v in cluster_stats.items()},
                "silhouette_score": silhouette,
                "execution_time": time.time() - start_time,
                "advanced_metrics": advanced_metrics
            }
            
            api.set(f"algo_cache.kmeans_clustering.{interval}", result)
            logging.info(f"Computed enhanced K-Means clustering for {interval} in {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhanced K-Means clustering for {interval}: {e}")
            return {"kmeans_clusters": {}, "kmeans_centroids": [], "cluster_stats": {}} 
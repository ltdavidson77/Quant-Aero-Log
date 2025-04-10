import cupy as cp
import numpy as np
from typing import Dict, List, Any, Tuple
from algorithm_support import AlgorithmBase
import logging
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import time

class Algorithm(AlgorithmBase):
    def get_default_config(self) -> Dict:
        return {
            "base_metric": "Adj Close",
            "correlation_threshold": 0.7,
            "use_advanced_metrics": True,
            "pca_components": 3,
            "volatility_window": 20,
            "momentum_window": 10,
            "correlation_method": "hybrid",  # hybrid, pearson, spearman, kendall
            "centrality_weight": 0.3,
            "volatility_weight": 0.2,
            "momentum_weight": 0.2,
            "liquidity_weight": 0.1,
            "max_workers": 4,
            "batch_size": 50,
            "use_gpu": True,
            "optimization_level": "high"  # high, medium, low
        }

    def get_supported_metrics(self) -> List[str]:
        return ["Adj Close", "Volume", "High", "Low", "Open", "Close"]

    def get_version(self) -> str:
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
            
            metrics[stock] = {
                "volatility": volatility,
                "momentum": momentum,
                "liquidity": liquidity,
                "returns": returns[-1] if len(returns) > 0 else 0.0
            }
        return metrics

    def _compute_correlation_matrix(self, data: Dict[str, Dict[str, float]], method: str = "hybrid") -> cp.ndarray:
        """Compute correlation matrix using multiple methods."""
        stocks = list(data.keys())
        N = len(stocks)
        
        if method == "hybrid":
            # Compute multiple correlation matrices
            pearson = cp.corrcoef(cp.array([cp.log1p(data[s].get("Adj Close", 0.0)) for s in stocks]))
            spearman = cp.array([[spearmanr([data[s1].get("Adj Close", 0.0) for _ in range(10)],
                                          [data[s2].get("Adj Close", 0.0) for _ in range(10)])[0]
                                for s2 in stocks] for s1 in stocks])
            kendall = cp.array([[kendalltau([data[s1].get("Adj Close", 0.0) for _ in range(10)],
                                          [data[s2].get("Adj Close", 0.0) for _ in range(10)])[0]
                               for s2 in stocks] for s1 in stocks])
            
            # Combine correlations with weights
            corr_matrix = 0.4 * pearson + 0.3 * spearman + 0.3 * kendall
        else:
            # Use single method
            if method == "pearson":
                corr_matrix = cp.corrcoef(cp.array([cp.log1p(data[s].get("Adj Close", 0.0)) for s in stocks]))
            elif method == "spearman":
                corr_matrix = cp.array([[spearmanr([data[s1].get("Adj Close", 0.0) for _ in range(10)],
                                                 [data[s2].get("Adj Close", 0.0) for _ in range(10)])[0]
                                      for s2 in stocks] for s1 in stocks])
            else:  # kendall
                corr_matrix = cp.array([[kendalltau([data[s1].get("Adj Close", 0.0) for _ in range(10)],
                                                  [data[s2].get("Adj Close", 0.0) for _ in range(10)])[0]
                                      for s2 in stocks] for s1 in stocks])
        
        return corr_matrix

    def _apply_pca(self, data: Dict[str, Dict[str, float]], n_components: int) -> Dict[str, np.ndarray]:
        """Apply PCA for dimensionality reduction."""
        features = []
        stocks = []
        for stock, values in data.items():
            features.append([
                values.get("Adj Close", 0.0),
                values.get("Volume", 0.0),
                values.get("High", 0.0),
                values.get("Low", 0.0)
            ])
            stocks.append(stock)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(scaled_features)
        
        return {stock: features for stock, features in zip(stocks, reduced_features)}

    def _compute_network_metrics(self, corr_matrix: cp.ndarray, stocks: List[str]) -> Dict[str, float]:
        """Compute network-based metrics."""
        G = nx.Graph()
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if i < j and corr_matrix[i, j] > 0.5:
                    G.add_edge(stock1, stock2, weight=float(corr_matrix[i, j]))
        
        centrality = nx.betweenness_centrality(G, weight='weight')
        clustering = nx.clustering(G, weight='weight')
        
        return {
            "centrality": centrality,
            "clustering": clustering
        }

    def _optimize_paths(self, distances: cp.ndarray, stocks: List[str], 
                       network_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Optimize paths using advanced techniques."""
        N = len(stocks)
        paths = {}
        
        def process_stock(start_idx: int) -> Tuple[str, List[str]]:
            unvisited = set(range(N))
            dist = cp.full(N, cp.inf)
            dist[start_idx] = 0
            path = [stocks[start_idx]]
            
            while unvisited:
                current = min(unvisited, key=lambda x: dist[x])
                unvisited.remove(current)
                
                for neighbor in range(N):
                    if neighbor in unvisited:
                        # Enhanced distance calculation
                        base_dist = distances[current, neighbor]
                        centrality_factor = 1 - network_metrics["centrality"].get(stocks[neighbor], 0.0)
                        clustering_factor = 1 - network_metrics["clustering"].get(stocks[neighbor], 0.0)
                        
                        alt = dist[current] + base_dist * (1 + centrality_factor + clustering_factor)
                        if alt < dist[neighbor]:
                            dist[neighbor] = alt
                            path.append(stocks[neighbor])
            
            return stocks[start_idx], path

        with ThreadPoolExecutor(max_workers=self.get_default_config()["max_workers"]) as executor:
            futures = [executor.submit(process_stock, i) for i in range(N)]
            for future in futures:
                stock, path = future.result()
                paths[stock] = path

        return paths

    def run(self, api: 'AnalysisAPI', data: Dict[str, Dict[str, float]], 
            interval: str, config: Dict, support_results: Dict[str, Any] = None) -> Dict[str, Any]:
        stocks = list(data.keys())
        N = len(stocks)
        if N < 2:
            return {"correlation_paths": {}}

        try:
            start_time = time.time()
            
            # Get configuration
            config = {**self.get_default_config(), **config}
            
            # Compute advanced metrics
            advanced_metrics = self._compute_advanced_metrics(data, config["volatility_window"])
            
            # Compute correlation matrix
            corr_matrix = self._compute_correlation_matrix(data, config["correlation_method"])
            
            # Apply PCA for dimensionality reduction
            pca_features = self._apply_pca(data, config["pca_components"])
            
            # Compute network metrics
            network_metrics = self._compute_network_metrics(corr_matrix, stocks)
            
            # Compute distances with advanced adjustments
            distances = 1 - cp.abs(cp.sin(corr_matrix))
            distances = cp.where(distances < 1 - config["correlation_threshold"], distances, cp.inf)
            
            # Adjust distances with multiple factors
            for i, stock in enumerate(stocks):
                centrality_factor = network_metrics["centrality"].get(stock, 0.0) * config["centrality_weight"]
                volatility_factor = advanced_metrics[stock]["volatility"] * config["volatility_weight"]
                momentum_factor = advanced_metrics[stock]["momentum"] * config["momentum_weight"]
                liquidity_factor = advanced_metrics[stock]["liquidity"] * config["liquidity_weight"]
                
                total_factor = 1 + centrality_factor + volatility_factor + momentum_factor + liquidity_factor
                distances[i] = distances[i] * total_factor

            # Optimize paths using parallel processing
            paths = self._optimize_paths(distances, stocks, network_metrics)
            
            # Filter and sort paths
            filtered_paths = {k: v for k, v in paths.items() if len(v) > 1}
            sorted_paths = dict(sorted(filtered_paths.items(), 
                                     key=lambda x: len(x[1]), 
                                     reverse=True))

            result = {
                "correlation_paths": sorted_paths,
                "execution_time": time.time() - start_time,
                "network_metrics": network_metrics,
                "advanced_metrics": advanced_metrics
            }
            
            api.set(f"algo_cache.dijkstra_correlation.{interval}", result)
            logging.info(f"Computed enhanced Dijkstra correlation for {interval} in {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhanced Dijkstra correlation for {interval}: {e}")
            return {"correlation_paths": {}} 
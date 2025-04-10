import cupy as cp
import numpy as np
from heapq import heappush, heappop
from typing import Dict, List, Any, Tuple, Optional
import logging
from .algorithm_support import AlgorithmBase
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from enum import Enum

class OptimizationLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Node:
    stock: str
    g_score: float
    f_score: float
    parent: Optional['Node'] = None

class Algorithm(AlgorithmBase):
    """A* algorithm implementation for stock prioritization with mesh topology integration."""
    
    def get_default_config(self) -> Dict:
        """Get default configuration for the algorithm."""
        return {
            "goal_metric": "Vol_1d",
            "weight_factor": 0.5,
            "use_advanced_metrics": True,
            "pca_components": 3,
            "volatility_window": 20,
            "momentum_window": 10,
            "correlation_method": "hybrid",
            "centrality_weight": 0.3,
            "volatility_weight": 0.2,
            "momentum_weight": 0.2,
            "liquidity_weight": 0.1,
            "max_workers": 4,
            "batch_size": 50,
            "use_gpu": True,
            "optimization_level": OptimizationLevel.HIGH,
            "heuristic_weight": 1.0,
            "risk_adjustment": True,
            "market_regime": "normal",  # normal, volatile, trending
            "time_horizon": "short",    # short, medium, long
            "sector_weighting": True,
            "liquidity_threshold": 0.1,
            "max_path_length": 10
        }

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return ["Adj Close", "Vol_1d", "Volume", "High", "Low", "Open", "Close"]

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

    def _compute_heuristic(self, current: str, goal_metric: str, data: Dict[str, Dict[str, float]], 
                          advanced_metrics: Dict[str, Dict[str, float]]) -> float:
        """Compute heuristic value for A* algorithm."""
        current_value = data[current].get(goal_metric, 0.0)
        max_value = max(data[s].get(goal_metric, 0.0) for s in data)
        
        # Base heuristic
        heuristic = 1 - (current_value / max_value) if max_value > 0 else 1
        
        # Adjust for market regime
        if self.get_default_config()["market_regime"] == "volatile":
            heuristic *= (1 + advanced_metrics[current]["volatility"])
        elif self.get_default_config()["market_regime"] == "trending":
            heuristic *= (1 - abs(advanced_metrics[current]["trend_strength"]))
        
        # Adjust for risk
        if self.get_default_config()["risk_adjustment"]:
            heuristic *= (1 + advanced_metrics[current]["risk_adjusted_returns"])
        
        return heuristic

    def _get_neighbors(self, current: str, data: Dict[str, Dict[str, float]], 
                      corr_matrix: cp.ndarray, stocks: List[str]) -> List[str]:
        """Get valid neighbors for the current node."""
        current_idx = stocks.index(current)
        correlations = corr_matrix[current_idx]
        
        # Filter neighbors based on correlation threshold
        threshold = 1 - self.get_default_config()["weight_factor"]
        neighbors = [stocks[i] for i in range(len(stocks)) 
                    if i != current_idx and correlations[i] > threshold]
        
        return neighbors

    def _reconstruct_path(self, current: Node) -> List[str]:
        """Reconstruct the path from the goal node."""
        path = []
        while current is not None:
            path.append(current.stock)
            current = current.parent
        return path[::-1]

    def _a_star_search(self, start: str, data: Dict[str, Dict[str, float]], 
                      corr_matrix: cp.ndarray, advanced_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Perform A* search with advanced features."""
        stocks = list(data.keys())
        config = self.get_default_config()
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Initialize start node
        start_node = Node(
            stock=start,
            g_score=0.0,
            f_score=self._compute_heuristic(start, config["goal_metric"], data, advanced_metrics)
        )
        heappush(open_set, (start_node.f_score, id(start_node), start_node))
        
        # Initialize g_scores
        g_scores = {stock: float('inf') for stock in stocks}
        g_scores[start] = 0.0
        
        while open_set:
            current_f_score, _, current_node = heappop(open_set)
            
            if current_node.stock in closed_set:
                continue
                
            closed_set.add(current_node.stock)
            
            # Check if we've reached the maximum path length
            if len(self._reconstruct_path(current_node)) >= config["max_path_length"]:
                return self._reconstruct_path(current_node)
            
            # Get neighbors
            neighbors = self._get_neighbors(current_node.stock, data, corr_matrix, stocks)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Compute tentative g_score
                tentative_g_score = g_scores[current_node.stock] + \
                    (1 - corr_matrix[stocks.index(current_node.stock), stocks.index(neighbor)])
                
                if tentative_g_score < g_scores[neighbor]:
                    # Update g_score and create new node
                    g_scores[neighbor] = tentative_g_score
                    neighbor_node = Node(
                        stock=neighbor,
                        g_score=tentative_g_score,
                        f_score=tentative_g_score + self._compute_heuristic(
                            neighbor, config["goal_metric"], data, advanced_metrics
                        ),
                        parent=current_node
                    )
                    heappush(open_set, (neighbor_node.f_score, id(neighbor_node), neighbor_node))
        
        return self._reconstruct_path(current_node)

    def run(self, api: 'AnalysisAPI', data: Dict[str, Dict[str, float]], 
            interval: str, config: Dict, support_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run A* algorithm for stock prioritization.
        
        Args:
            api: Analysis API instance
            data: Stock data dictionary
            interval: Time interval
            config: Algorithm configuration
            support_results: Additional support results
            
        Returns:
            Dictionary containing prioritized stocks and their scores
        """
        stocks = list(data.keys())
        if len(stocks) < 2:
            return {"astar_priority": [], "astar_scores": []}

        try:
            start_time = time.time()
            
            # Get configuration
            config = {**self.get_default_config(), **config}
            
            # Compute advanced metrics
            advanced_metrics = self._compute_advanced_metrics(data, config["volatility_window"])
            
            # Compute correlation matrix
            corr_matrix = self._compute_correlation_matrix(data, config["correlation_method"])
            
            # Find optimal paths for each stock
            paths = {}
            scores = {}
            
            for stock in stocks:
                path = self._a_star_search(stock, data, corr_matrix, advanced_metrics)
                if len(path) > 1:  # Only include paths with at least one step
                    paths[stock] = path
                    # Compute path score based on goal metric
                    score = sum(data[s].get(config["goal_metric"], 0.0) for s in path) / len(path)
                    scores[stock] = score
            
            # Sort stocks by their scores
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            priority = [stock for stock, _ in sorted_stocks]
            
            result = {
                "astar_priority": priority,
                "astar_scores": dict(sorted_stocks),
                "execution_time": time.time() - start_time,
                "advanced_metrics": advanced_metrics
            }
            
            api.set(f"algo_cache.astar_prioritization.{interval}", result)
            logging.info(f"Computed enhanced A* prioritization for {interval} in {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in enhanced A* prioritization for {interval}: {e}")
            return {"astar_priority": [], "astar_scores": []} 
import cupy as cp
import numpy as np
from typing import Dict, List, Any
from algorithm_support import AlgorithmBase
import logging

class Algorithm(AlgorithmBase):
    def get_default_config(self) -> Dict:
        return {"max_iter": 100, "temp": 1000.0, "cooling_rate": 0.95, "metric": "Adj Close"}

    def get_supported_metrics(self) -> List[str]:
        return ["Adj Close", "Vol_1d", "Volume"]

    def get_version(self) -> str:
        return "v2.0"

    def run(self, api: 'AnalysisAPI', data: Dict[str, Dict[str, float]], interval: str, config: Dict, support_results: Dict[str, Any] = None) -> Dict[str, Any]:
        stocks = list(data.keys())
        N = len(stocks)
        if N < 2:
            return {"sa_portfolio": [], "sa_score": 0.0}

        try:
            # Integrate ultra mesh topology
            mesh_result = api.get(f"mesh_results.{interval}", {})
            centrality = mesh_result.get("centrality", {})

            values = cp.array([cp.sin(cp.log1p(data[s].get(config["metric"], 0.0))) for s in stocks])
            # Adjust with mesh centrality
            for i, stock in enumerate(stocks):
                mesh_factor = centrality.get(stock, 0.0) * 0.06
                values[i] += mesh_factor

            current = cp.random.choice(N, size=min(5, N), replace=False)
            current_score = cp.sum(values[current])
            best = current.copy()
            best_score = current_score

            temp = config["temp"]
            for _ in range(config["max_iter"]):
                neighbor = current.copy()
                idx = np.random.randint(len(neighbor))
                neighbor[idx] = np.random.randint(N)
                neighbor_score = cp.sum(values[neighbor])

                if neighbor_score > current_score or cp.exp((neighbor_score - current_score) / temp) > cp.random.rand():
                    current = neighbor
                    current_score = neighbor_score
                    if current_score > best_score:
                        best = current
                        best_score = current_score
                temp *= config["cooling_rate"]

            result = {
                "sa_portfolio": [stocks[int(i)] for i in best],
                "sa_score": float(best_score)
            }
            api.set(f"algo_cache.simulated_annealing.{interval}", result)
            logging.info(f"Computed Simulated Annealing for {interval}")
            return result
        except Exception as e:
            logging.error(f"Error in Simulated Annealing for {interval}: {e}")
            return {"sa_portfolio": [], "sa_score": 0.0} 
# ==========================
# performance_monitor.py
# ==========================
# Performance monitoring utilities.

import time
import psutil
import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import gc

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    quantum_success_rate: float
    quantum_state_quality: float
    cache_hit_rate: float
    resource_utilization: float
    quantum_evolution_rate: float
    timestamp: datetime

class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        
    def update(self, execution_time: float, success: bool,
               quantum_success_rate: float = 0.0,
               quantum_state_quality: float = 0.0,
               cache_hit_rate: float = 0.0,
               resource_utilization: float = 0.0,
               quantum_evolution_rate: float = 0.0) -> None:
        """Update performance metrics."""
        try:
            # Get system metrics
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu_usage = psutil.cpu_percent()
            gpu_usage = None
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                
            # Create metrics object
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                quantum_success_rate=quantum_success_rate,
                quantum_state_quality=quantum_state_quality,
                cache_hit_rate=cache_hit_rate,
                resource_utilization=resource_utilization,
                quantum_evolution_rate=quantum_evolution_rate,
                timestamp=datetime.now()
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Maintain window size
            if len(self.metrics_history) > self.window_size:
                self.metrics_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            if not self.metrics_history:
                return {}
                
            latest = self.metrics_history[-1]
            return {
                'execution_time': latest.execution_time,
                'memory_usage': latest.memory_usage,
                'cpu_usage': latest.cpu_usage,
                'gpu_usage': latest.gpu_usage,
                'quantum_success_rate': latest.quantum_success_rate,
                'quantum_state_quality': latest.quantum_state_quality,
                'cache_hit_rate': latest.cache_hit_rate,
                'resource_utilization': latest.resource_utilization,
                'quantum_evolution_rate': latest.quantum_evolution_rate,
                'timestamp': latest.timestamp.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {}
            
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis over time."""
        try:
            if not self.metrics_history:
                return {}
                
            return {
                'avg_execution_time': sum(m.execution_time for m in self.metrics_history) / len(self.metrics_history),
                'max_execution_time': max(m.execution_time for m in self.metrics_history),
                'min_execution_time': min(m.execution_time for m in self.metrics_history),
                'avg_memory_usage': sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history),
                'avg_cpu_usage': sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history),
                'avg_gpu_usage': sum(m.gpu_usage for m in self.metrics_history if m.gpu_usage is not None) / len([m for m in self.metrics_history if m.gpu_usage is not None]) if any(m.gpu_usage is not None for m in self.metrics_history) else None,
                'avg_quantum_success_rate': sum(m.quantum_success_rate for m in self.metrics_history) / len(self.metrics_history),
                'avg_quantum_state_quality': sum(m.quantum_state_quality for m in self.metrics_history) / len(self.metrics_history),
                'avg_cache_hit_rate': sum(m.cache_hit_rate for m in self.metrics_history) / len(self.metrics_history),
                'avg_resource_utilization': sum(m.resource_utilization for m in self.metrics_history) / len(self.metrics_history),
                'avg_quantum_evolution_rate': sum(m.quantum_evolution_rate for m in self.metrics_history) / len(self.metrics_history)
            }
        except Exception as e:
            logger.error(f"Error getting performance analysis: {str(e)}")
            return {}
            
    def reset(self) -> None:
        """Reset performance monitoring."""
        self.metrics_history.clear()
        self.start_time = time.time() 
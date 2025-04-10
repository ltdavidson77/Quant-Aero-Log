# ==========================
# metrics.py
# ==========================
# Monitoring and metrics collection for the Quant-Aero-Log framework.

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from typing import Dict, Any
from config_env import ENV
from utils.logger import get_logger

logger = get_logger("metrics")

# Initialize Prometheus metrics
class Metrics:
    """Collection of Prometheus metrics for monitoring."""
    
    # Execution metrics
    execution_counter = Counter(
        'quant_aero_executions_total',
        'Total number of framework executions',
        ['environment', 'status']
    )
    
    execution_duration = Histogram(
        'quant_aero_execution_duration_seconds',
        'Duration of framework executions',
        ['environment'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )
    
    # Model metrics
    model_training_duration = Histogram(
        'quant_aero_model_training_duration_seconds',
        'Duration of model training',
        ['environment', 'model_type'],
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
    )
    
    model_accuracy = Gauge(
        'quant_aero_model_accuracy',
        'Model accuracy metric',
        ['environment', 'model_type', 'horizon']
    )
    
    # Database metrics
    db_connections = Gauge(
        'quant_aero_db_connections',
        'Number of active database connections',
        ['environment']
    )
    
    db_query_duration = Histogram(
        'quant_aero_db_query_duration_seconds',
        'Duration of database queries',
        ['environment', 'operation'],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    # API metrics
    api_request_duration = Histogram(
        'quant_aero_api_request_duration_seconds',
        'Duration of API requests',
        ['environment', 'endpoint'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    api_errors = Counter(
        'quant_aero_api_errors_total',
        'Total number of API errors',
        ['environment', 'endpoint', 'error_type']
    )
    
    # Signal metrics
    signal_generation_duration = Histogram(
        'quant_aero_signal_generation_duration_seconds',
        'Duration of signal generation',
        ['environment', 'signal_type'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    signal_quality = Gauge(
        'quant_aero_signal_quality',
        'Quality metric for generated signals',
        ['environment', 'signal_type']
    )

def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info("metrics_server_started", port=port)
    except Exception as e:
        logger.error("metrics_server_start_failed", error=str(e))
        raise

def record_execution_metrics(metrics: Dict[str, Any]) -> None:
    """Record execution metrics."""
    env = ENV.value
    Metrics.execution_counter.labels(environment=env, status='success').inc()
    Metrics.execution_duration.labels(environment=env).observe(
        metrics.get('duration_seconds', 0)
    )

def record_model_metrics(metrics: Dict[str, Any]) -> None:
    """Record model training and performance metrics."""
    env = ENV.value
    model_type = metrics.get('model_type', 'unknown')
    horizon = metrics.get('horizon', 'unknown')
    
    Metrics.model_training_duration.labels(
        environment=env,
        model_type=model_type
    ).observe(metrics.get('training_duration_seconds', 0))
    
    Metrics.model_accuracy.labels(
        environment=env,
        model_type=model_type,
        horizon=horizon
    ).set(metrics.get('accuracy', 0))

def record_database_metrics(metrics: Dict[str, Any]) -> None:
    """Record database operation metrics."""
    env = ENV.value
    operation = metrics.get('operation', 'unknown')
    
    Metrics.db_connections.labels(environment=env).set(
        metrics.get('connection_count', 0)
    )
    
    Metrics.db_query_duration.labels(
        environment=env,
        operation=operation
    ).observe(metrics.get('query_duration_seconds', 0))

def record_api_metrics(metrics: Dict[str, Any]) -> None:
    """Record API request metrics."""
    env = ENV.value
    endpoint = metrics.get('endpoint', 'unknown')
    
    Metrics.api_request_duration.labels(
        environment=env,
        endpoint=endpoint
    ).observe(metrics.get('request_duration_seconds', 0))
    
    if metrics.get('error'):
        Metrics.api_errors.labels(
            environment=env,
            endpoint=endpoint,
            error_type=metrics.get('error_type', 'unknown')
        ).inc()

def record_signal_metrics(metrics: Dict[str, Any]) -> None:
    """Record signal generation metrics."""
    env = ENV.value
    signal_type = metrics.get('signal_type', 'unknown')
    
    Metrics.signal_generation_duration.labels(
        environment=env,
        signal_type=signal_type
    ).observe(metrics.get('generation_duration_seconds', 0))
    
    Metrics.signal_quality.labels(
        environment=env,
        signal_type=signal_type
    ).set(metrics.get('quality', 0)) 
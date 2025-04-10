# ==========================
# logger.py
# ==========================
# Centralized logging configuration for the Quant-Aero-Log framework.

import logging
import structlog
from typing import Any, Dict
from pathlib import Path
from config_env import LOG_CONFIG, PATHS, ENV

def setup_logging() -> None:
    """Configure structured logging for the application."""
    # Ensure log directory exists
    log_dir = Path(PATHS["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=LOG_CONFIG["level"].value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_dir / f"quant_aero_{ENV.value.lower()}.log",
                maxBytes=LOG_CONFIG["max_size_mb"] * 1024 * 1024,
                backupCount=LOG_CONFIG["backup_count"]
            )
        ]
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)

class LogContext:
    """Context manager for logging with additional context."""
    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any):
        self.logger = logger
        self.context = kwargs
        
    def __enter__(self) -> structlog.BoundLogger:
        return self.logger.bind(**self.context)
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

def log_execution_metrics(metrics: Dict[str, Any]) -> None:
    """Log execution metrics in a structured format."""
    logger = get_logger("execution")
    with LogContext(logger, 
                   environment=ENV.value,
                   execution_id=metrics.get("execution_id", "unknown")):
        logger.info("execution_metrics", **metrics)

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log errors with full context."""
    logger = get_logger("error")
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "environment": ENV.value
    }
    if context:
        error_context.update(context)
    logger.error("error_occurred", **error_context)

def log_performance_metrics(metrics: Dict[str, Any]) -> None:
    """Log performance metrics for monitoring."""
    logger = get_logger("performance")
    with LogContext(logger, environment=ENV.value):
        logger.info("performance_metrics", **metrics) 
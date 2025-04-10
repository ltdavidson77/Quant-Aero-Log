# ==========================
# logger.py
# ==========================
# Centralized logging configuration for the Quant-Aero-Log framework.

import logging
import structlog
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import json
import traceback
from functools import wraps
import time
import threading
from queue import Queue
import gzip
import os
from config_env import LOG_CONFIG, PATHS, ENV

class LogQueue:
    """Thread-safe queue for asynchronous logging."""
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        
    def _process_queue(self):
        while True:
            log_entry = self.queue.get()
            if log_entry is None:
                break
            logger, level, msg, kwargs = log_entry
            logger.log(level, msg, **kwargs)
            self.queue.task_done()
            
    def put(self, logger, level, msg, **kwargs):
        self.queue.put((logger, level, msg, kwargs))
        
    def stop(self):
        self.queue.put(None)
        self.thread.join()

class LogRotator:
    """Handles log file rotation with compression."""
    def __init__(self, max_size_mb: int, backup_count: int):
        self.max_size = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
    def should_rotate(self, file_path: Path) -> bool:
        return file_path.exists() and file_path.stat().st_size >= self.max_size
        
    def rotate(self, file_path: Path) -> None:
        if not self.should_rotate(file_path):
            return
            
        # Compress and rotate existing logs
        for i in range(self.backup_count - 1, 0, -1):
            src = file_path.parent / f"{file_path.stem}.{i}.gz"
            dst = file_path.parent / f"{file_path.stem}.{i+1}.gz"
            if src.exists():
                src.rename(dst)
                
        # Compress current log
        with open(file_path, 'rb') as f_in:
            with gzip.open(f"{file_path}.1.gz", 'wb') as f_out:
                f_out.writelines(f_in)
                
        # Clear current log
        file_path.unlink()
        file_path.touch()

def setup_logging() -> None:
    """Configure structured logging for the application."""
    try:
        # Ensure log directory exists
        log_dir = Path(PATHS["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log rotator
        rotator = LogRotator(
            max_size_mb=LOG_CONFIG["max_size_mb"],
            backup_count=LOG_CONFIG["backup_count"]
        )
        
        # Configure structlog with enhanced processors
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True
        )
        
        # Configure standard logging with rotation
        log_file = log_dir / f"quant_aero_{ENV.value.lower()}.log"
        logging.basicConfig(
            level=LOG_CONFIG["level"].value,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    log_file,
                    maxBytes=LOG_CONFIG["max_size_mb"] * 1024 * 1024,
                    backupCount=LOG_CONFIG["backup_count"]
                )
            ]
        )
        
        # Initialize async logging queue
        global log_queue
        log_queue = LogQueue()
        
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        raise

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance with enhanced features."""
    logger = structlog.get_logger(name)
    
    # Add correlation ID for distributed tracing
    correlation_id = os.environ.get('CORRELATION_ID', '')
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
        
    return logger

class LogContext:
    """Enhanced context manager for logging with additional context."""
    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any):
        self.logger = logger
        self.context = kwargs
        self.start_time = time.time()
        
    def __enter__(self) -> structlog.BoundLogger:
        return self.logger.bind(**self.context)
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(
                "context_exit_with_error",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                duration=duration,
                traceback=traceback.format_exc()
            )
        else:
            self.logger.info("context_exit", duration=duration)

def log_execution_metrics(metrics: Dict[str, Any]) -> None:
    """Enhanced execution metrics logging with async support."""
    logger = get_logger("execution")
    metrics.update({
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENV.value
    })
    
    # Use async logging for better performance
    log_queue.put(
        logger,
        logging.INFO,
        "execution_metrics",
        **metrics
    )

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Enhanced error logging with detailed context."""
    logger = get_logger("error")
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "environment": ENV.value,
        "timestamp": datetime.utcnow().isoformat(),
        "traceback": traceback.format_exc()
    }
    if context:
        error_context.update(context)
        
    # Use async logging for better performance
    log_queue.put(
        logger,
        logging.ERROR,
        "error_occurred",
        **error_context
    )

def log_performance_metrics(metrics: Dict[str, Any]) -> None:
    """Enhanced performance metrics logging with async support."""
    logger = get_logger("performance")
    metrics.update({
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENV.value
    })
    
    # Use async logging for better performance
    log_queue.put(
        logger,
        logging.INFO,
        "performance_metrics",
        **metrics
    )

def log_metric(name: str, value: float, tags: Dict[str, str] = None) -> None:
    """Log a single metric with tags."""
    logger = get_logger("metrics")
    metric_data = {
        "metric_name": name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENV.value
    }
    if tags:
        metric_data["tags"] = tags
        
    log_queue.put(
        logger,
        logging.INFO,
        "metric",
        **metric_data
    )

def log_trace(span_name: str, **kwargs) -> None:
    """Log a trace span for distributed tracing."""
    logger = get_logger("trace")
    trace_data = {
        "span_name": span_name,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENV.value,
        **kwargs
    }
    
    log_queue.put(
        logger,
        logging.INFO,
        "trace_span",
        **trace_data
    )

def timed(logger: Optional[structlog.BoundLogger] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.info(
                        "function_timing",
                        function_name=func.__name__,
                        duration=duration
                    )
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(
                        "function_timing_error",
                        function_name=func.__name__,
                        duration=duration,
                        error=str(e)
                    )
                raise
        return wrapper
    return decorator 
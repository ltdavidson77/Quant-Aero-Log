# ==========================
# debug_main.py
# ==========================
# Helper script for running main application in debug mode.

import os
import sys
import logging
import cProfile
import pstats
import traceback
import tracemalloc
import psutil
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps

# Configure logging for debug mode
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'debug_main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Set debug environment variables
os.environ['DEBUG_MODE'] = 'true'
os.environ['PYTHONPATH'] = str(project_root)
os.environ['LOG_LEVEL'] = 'DEBUG'

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        logging.info(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
        
        return result
    return wrapper

def async_performance_monitor(func):
    """Decorator to monitor async function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = await func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        logging.info(f"Async function {func.__name__} took {end_time - start_time:.2f} seconds")
        logging.info(f"Memory usage: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
        
        return result
    return wrapper

def setup_debug_environment():
    """Setup debug environment and logging."""
    # Enable detailed exception reporting
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: (
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    )
    
    # Enable detailed traceback
    traceback.print_exc = lambda *args, **kwargs: (
        logging.error("Traceback", exc_info=sys.exc_info())
    )
    
    # Enable memory tracking
    if os.environ.get('ENABLE_MEMORY_TRACKING') == 'true':
        tracemalloc.start()
        logging.info("Memory tracking enabled")
    
    # Enable performance monitoring
    if os.environ.get('ENABLE_PERFORMANCE_MONITORING') == 'true':
        logging.info("Performance monitoring enabled")

def run_with_profiling(func, *args, **kwargs) -> Optional[pstats.Stats]:
    """Run a function with profiling enabled."""
    if os.environ.get('ENABLE_PROFILING') == 'true':
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Save profiling results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats.dump_stats(f'profile_main_{timestamp}.prof')
            
            # Print top 20 functions
            stats.print_stats(20)
            return stats
    else:
        return func(*args, **kwargs)

def log_memory_usage():
    """Log current memory usage if tracking is enabled."""
    if os.environ.get('ENABLE_MEMORY_TRACKING') == 'true':
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logging.info("Memory usage top 10:")
        for stat in top_stats[:10]:
            logging.info(f"{stat.count} blocks: {stat.size / 1024 / 1024:.2f} MB")
            logging.info(f"  {stat.traceback.format()[-1]}")

def log_system_resources():
    """Log system resource usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    logging.info(f"CPU Usage: {cpu_percent}%")
    logging.info(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    logging.info(f"Virtual Memory: {memory_info.vms / 1024 / 1024:.2f} MB")

@performance_monitor
def main():
    """Run main application in debug mode."""
    try:
        setup_debug_environment()
        logging.info("Starting main application in debug mode")
        
        # Log initial system resources
        log_system_resources()
        
        # Import and run main application
        from main import main as app_main
        
        # Run with profiling if enabled
        stats = run_with_profiling(app_main)
        
        # Log final system resources
        log_system_resources()
        log_memory_usage()
        
        if stats is None or stats == 0:
            logging.info("Application completed successfully")
        else:
            logging.error(f"Application failed with exit code: {stats}")
            
    except Exception as e:
        logging.error(f"Error running application: {str(e)}")
        logging.error("Full traceback:", exc_info=True)
        raise

if __name__ == '__main__':
    main() 
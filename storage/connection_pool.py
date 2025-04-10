# ==========================
# storage/connection_pool.py
# ==========================
# Database connection pool management.

import psycopg2
from psycopg2 import pool
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import time
import psutil
from db_connector import DatabaseError, QueryError
from utils.logger import get_logger

logger = get_logger("connection_pool")

class ConnectionPool:
    """Manages database connection pooling and optimization."""
    
    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 20,
        idle_timeout: int = 300,  # seconds
        max_lifetime: int = 3600,  # seconds
        stats_dir: str = "pool_stats"
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self.max_lifetime = max_lifetime
        self.stats_dir = Path(stats_dir)
        self.stats_file = self.stats_dir / "pool_stats.json"
        self._setup_stats_directory()
        self._load_stats()
        
        # Initialize connection pool
        self.pool = self._create_pool()
        self._start_monitoring()
    
    def _setup_stats_directory(self) -> None:
        """Setup statistics directory structure."""
        try:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Pool stats directory created at {self.stats_dir}")
        except Exception as e:
            logger.error(f"Failed to setup stats directory: {str(e)}")
            raise
    
    def _load_stats(self) -> None:
        """Load pool statistics from file."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                self.stats = {
                    'total_connections': 0,
                    'active_connections': 0,
                    'idle_connections': 0,
                    'failed_connections': 0,
                    'connection_history': [],
                    'performance_metrics': {
                        'avg_wait_time': 0,
                        'max_wait_time': 0,
                        'total_wait_time': 0,
                        'wait_count': 0
                    }
                }
                self._save_stats()
        except Exception as e:
            logger.error(f"Failed to load pool stats: {str(e)}")
            raise
    
    def _save_stats(self) -> None:
        """Save pool statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pool stats: {str(e)}")
            raise
    
    def _create_pool(self) -> pool.ThreadedConnectionPool:
        """Create a new connection pool."""
        try:
            return pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=os.getenv("PG_HOST", "localhost"),
                port=os.getenv("PG_PORT", "5432"),
                user=os.getenv("PG_USER", "postgres"),
                password=os.getenv("PG_PASSWORD", ""),
                database=os.getenv("PG_DB", "quantdata")
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {str(e)}")
            raise
    
    def _start_monitoring(self) -> None:
        """Start connection pool monitoring."""
        try:
            self.monitor_thread = threading.Thread(
                target=self._monitor_pool,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("Connection pool monitoring started")
        except Exception as e:
            logger.error(f"Failed to start pool monitoring: {str(e)}")
            raise
    
    def _monitor_pool(self) -> None:
        """Monitor connection pool health and performance."""
        while True:
            try:
                # Check pool status
                self._check_pool_health()
                
                # Clean up idle connections
                self._cleanup_idle_connections()
                
                # Update statistics
                self._update_pool_stats()
                
                # Sleep for monitoring interval
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Pool monitoring error: {str(e)}")
                time.sleep(60)
    
    def _check_pool_health(self) -> None:
        """Check connection pool health."""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80 or memory.percent > 85:
                logger.warning(
                    f"High system resource usage (CPU: {cpu_percent}%, Memory: {memory.percent}%), "
                    "may affect connection pool performance"
                )
            
            # Check pool status
            if self.pool.closed:
                logger.error("Connection pool is closed, attempting to recreate")
                self.pool = self._create_pool()
            
        except Exception as e:
            logger.error(f"Failed to check pool health: {str(e)}")
            raise
    
    def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        try:
            current_time = datetime.now()
            
            for conn in self.pool._pool:
                if conn.closed:
                    continue
                
                # Check connection age
                if (current_time - conn._created).total_seconds() > self.max_lifetime:
                    logger.info(f"Closing old connection (age: {(current_time - conn._created).total_seconds()}s)")
                    conn.close()
                    continue
                
                # Check idle time
                if (current_time - conn._last_used).total_seconds() > self.idle_timeout:
                    logger.info(f"Closing idle connection (idle: {(current_time - conn._last_used).total_seconds()}s)")
                    conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cleanup idle connections: {str(e)}")
            raise
    
    def _update_pool_stats(self) -> None:
        """Update pool statistics."""
        try:
            self.stats['total_connections'] = len(self.pool._pool)
            self.stats['active_connections'] = sum(1 for conn in self.pool._pool if not conn.closed)
            self.stats['idle_connections'] = sum(
                1 for conn in self.pool._pool
                if not conn.closed and
                (datetime.now() - conn._last_used).total_seconds() > 60
            )
            
            self._save_stats()
            
        except Exception as e:
            logger.error(f"Failed to update pool stats: {str(e)}")
            raise
    
    def get_connection(self) -> psycopg2.extensions.connection:
        """
        Get a connection from the pool.
        
        Returns:
            psycopg2 connection object
        """
        try:
            start_time = datetime.now()
            
            # Get connection from pool
            conn = self.pool.getconn()
            
            # Update connection stats
            wait_time = (datetime.now() - start_time).total_seconds()
            self.stats['performance_metrics']['total_wait_time'] += wait_time
            self.stats['performance_metrics']['wait_count'] += 1
            self.stats['performance_metrics']['avg_wait_time'] = (
                self.stats['performance_metrics']['total_wait_time'] /
                self.stats['performance_metrics']['wait_count']
            )
            self.stats['performance_metrics']['max_wait_time'] = max(
                self.stats['performance_metrics']['max_wait_time'],
                wait_time
            )
            
            # Update connection history
            self.stats['connection_history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'get',
                'wait_time': wait_time
            })
            
            self._save_stats()
            
            return conn
            
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {str(e)}")
            self.stats['failed_connections'] += 1
            self._save_stats()
            raise
    
    def return_connection(self, conn: psycopg2.extensions.connection) -> None:
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection to return
        """
        try:
            # Update connection stats
            self.stats['connection_history'].append({
                'timestamp': datetime.now().isoformat(),
                'action': 'return',
                'connection_time': (datetime.now() - conn._created).total_seconds()
            })
            
            # Return connection to pool
            self.pool.putconn(conn)
            
            self._save_stats()
            
        except Exception as e:
            logger.error(f"Failed to return connection to pool: {str(e)}")
            raise
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        try:
            self.pool.closeall()
            logger.info("All connections closed")
        except Exception as e:
            logger.error(f"Failed to close connections: {str(e)}")
            raise
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        try:
            return {
                'total_connections': self.stats['total_connections'],
                'active_connections': self.stats['active_connections'],
                'idle_connections': self.stats['idle_connections'],
                'failed_connections': self.stats['failed_connections'],
                'performance_metrics': self.stats['performance_metrics'],
                'recent_history': self.stats['connection_history'][-10:]
            }
        except Exception as e:
            logger.error(f"Failed to get pool stats: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Initialize connection pool
        pool = ConnectionPool(
            min_connections=1,
            max_connections=20,
            idle_timeout=300,
            max_lifetime=3600,
            stats_dir="pool_stats"
        )
        
        # Get connection
        conn = pool.get_connection()
        
        try:
            # Use connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                print("Test query result:", result)
        
        finally:
            # Return connection
            pool.return_connection(conn)
        
        # Get pool statistics
        stats = pool.get_pool_stats()
        print("\nPool Statistics:", json.dumps(stats, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        pool.close_all() 
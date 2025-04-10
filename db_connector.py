# ==========================
# db_connector.py
# ==========================
# Enhanced PostgreSQL connection management with pooling, monitoring, and robust error handling.

import psycopg2
from psycopg2 import pool, errors
from psycopg2.extras import execute_values
import os
import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from config_env import PG_DB, PG_USER, PG_PASS, PG_HOST, PG_PORT, DEBUG_MODE

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """Tracks connection performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    avg_connection_time: float = 0.0
    total_query_time: float = 0.0
    total_queries: int = 0

class DatabaseError(Exception):
    """Base exception for database-related errors"""
    pass

class ConnectionError(DatabaseError):
    """Raised when connection to database fails"""
    pass

class QueryError(DatabaseError):
    """Raised when a query execution fails"""
    pass

class ConnectionPool:
    """Thread-safe connection pool with monitoring"""
    
    def __init__(self, min_conn: int = 1, max_conn: int = 10):
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self.metrics = ConnectionMetrics()
        self._validate_config()
        self._initialize_pool()
    
    def _validate_config(self) -> None:
        """Validate database configuration"""
        required_vars = {
            'PG_DB': PG_DB,
            'PG_USER': PG_USER,
            'PG_PASS': PG_PASS,
            'PG_HOST': PG_HOST,
            'PG_PORT': PG_PORT
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ConnectionError(f"Missing required database configuration: {', '.join(missing_vars)}")
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool"""
        try:
            self.pool = pool.ThreadedConnectionPool(
                minconn=self.min_conn,
                maxconn=self.max_conn,
                dbname=PG_DB,
                user=PG_USER,
                password=PG_PASS,
                host=PG_HOST,
                port=PG_PORT,
                connect_timeout=10
            )
            logger.info("Connection pool initialized successfully")
        except errors.OperationalError as e:
            raise ConnectionError(f"Failed to initialize connection pool: {str(e)}")
    
    @contextmanager
    def get_connection(self) -> psycopg2.extensions.connection:
        """Get a connection from the pool with context management"""
        conn = None
        start_time = time.time()
        
        try:
            conn = self.pool.getconn()
            self.metrics.active_connections += 1
            self.metrics.total_connections += 1
            connection_time = time.time() - start_time
            self.metrics.avg_connection_time = (
                (self.metrics.avg_connection_time * (self.metrics.total_connections - 1) + connection_time)
                / self.metrics.total_connections
            )
            
            # Validate connection
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
            
            yield conn
            
        except errors.OperationalError as e:
            self.metrics.failed_connections += 1
            raise ConnectionError(f"Failed to get connection: {str(e)}")
        except Exception as e:
            raise DatabaseError(f"Unexpected error: {str(e)}")
        finally:
            if conn:
                self.pool.putconn(conn)
                self.metrics.active_connections -= 1
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query with performance monitoring"""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    result = cur.fetchall()
                    
                    query_time = time.time() - start_time
                    self.metrics.total_query_time += query_time
                    self.metrics.total_queries += 1
                    
                    if DEBUG_MODE:
                        logger.info(f"Query executed in {query_time:.3f}s: {query[:100]}...")
                    
                    return result
        except errors.Error as e:
            raise QueryError(f"Query execution failed: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current connection metrics"""
        return {
            'total_connections': self.metrics.total_connections,
            'active_connections': self.metrics.active_connections,
            'failed_connections': self.metrics.failed_connections,
            'avg_connection_time': self.metrics.avg_connection_time,
            'avg_query_time': self.metrics.total_query_time / self.metrics.total_queries if self.metrics.total_queries > 0 else 0,
            'total_queries': self.metrics.total_queries
        }
    
    def close(self) -> None:
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed")

# Global connection pool instance
db_pool: Optional[ConnectionPool] = None

def initialize_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    """Initialize the global connection pool"""
    global db_pool
    if not db_pool:
        db_pool = ConnectionPool(min_conn, max_conn)

def get_connection() -> psycopg2.extensions.connection:
    """Get a connection from the global pool"""
    if not db_pool:
        initialize_pool()
    return db_pool.get_connection()

def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a query using the global pool"""
    if not db_pool:
        initialize_pool()
    return db_pool.execute_query(query, params)

def get_metrics() -> Dict[str, Any]:
    """Get metrics from the global pool"""
    if not db_pool:
        initialize_pool()
    return db_pool.get_metrics()

def close_pool() -> None:
    """Close the global connection pool"""
    global db_pool
    if db_pool:
        db_pool.close()
        db_pool = None

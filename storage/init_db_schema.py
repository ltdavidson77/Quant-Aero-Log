# ==========================
# storage/init_db_schema.py
# ==========================
# Enhanced PostgreSQL schema initialization with optimization features.

from sqlalchemy import create_engine, text, MetaData, Table, Column, Index
from sqlalchemy.dialects.postgresql import TIMESTAMP, DOUBLE_PRECISION
from typing import List, Dict, Any
import os
from datetime import datetime, timedelta
from db_connector import execute_query, DatabaseError, QueryError
from utils.logger import get_logger

logger = get_logger("database_schema")

class DatabaseSchema:
    """Manages database schema initialization and optimization."""
    
    def __init__(self, table: str = "signal_snapshots"):
        self.table = table
        self.partition_interval = '1 month'  # Default partition interval
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = {
            'PG_USER': os.getenv("PG_USER"),
            'PG_PASSWORD': os.getenv("PG_PASSWORD"),
            'PG_HOST': os.getenv("PG_HOST"),
            'PG_DB': os.getenv("PG_DB"),
            'PG_PORT': os.getenv("PG_PORT")
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise DatabaseError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _create_partitioned_table(self) -> None:
        """Create a partitioned table with proper structure."""
        try:
            # Create parent table
            query = f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    {', '.join([f'log_{r}m DOUBLE PRECISION, alg_{r}m DOUBLE PRECISION' for r in [1,5,10,60,180,300,600,1440,2880,10080,43200]])[:-1]},
                    CONSTRAINT {self.table}_pkey PRIMARY KEY (timestamp)
                ) PARTITION BY RANGE (timestamp);
            """
            execute_query(query)
            logger.info(f"Created parent table {self.table}")
            
            # Create initial partitions
            self._create_initial_partitions()
            
        except DatabaseError as e:
            logger.error(f"Failed to create partitioned table: {str(e)}")
            raise
    
    def _create_initial_partitions(self) -> None:
        """Create initial partitions for the current and next month."""
        try:
            current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = (current_month + timedelta(days=32)).replace(day=1)
            
            for start_date in [current_month, next_month]:
                end_date = (start_date + timedelta(days=32)).replace(day=1)
                partition_name = f"{self.table}_{start_date.strftime('%Y%m')}"
                
                query = f"""
                    CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF {self.table}
                    FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}');
                """
                execute_query(query)
                logger.info(f"Created partition {partition_name}")
                
        except DatabaseError as e:
            logger.error(f"Failed to create initial partitions: {str(e)}")
            raise
    
    def _create_indexes(self) -> None:
        """Create optimized indexes for common query patterns."""
        try:
            # Create BRIN index for timestamp (good for time-series data)
            execute_query(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table}_timestamp_brin
                ON {self.table} USING BRIN (timestamp);
            """)
            
            # Create partial indexes for common time ranges
            for days in [1, 7, 30]:
                execute_query(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table}_recent_{days}d
                    ON {self.table} (timestamp)
                    WHERE timestamp >= NOW() - INTERVAL '{days} days';
                """)
            
            logger.info("Created optimized indexes")
            
        except DatabaseError as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise
    
    def _create_constraints(self) -> None:
        """Create data integrity constraints."""
        try:
            # Add check constraints for numeric ranges
            for timeframe in [1,5,10,60,180,300,600,1440,2880,10080,43200]:
                execute_query(f"""
                    ALTER TABLE {self.table}
                    ADD CONSTRAINT chk_log_{timeframe}m_range
                    CHECK (log_{timeframe}m BETWEEN -100 AND 100);
                    
                    ALTER TABLE {self.table}
                    ADD CONSTRAINT chk_alg_{timeframe}m_range
                    CHECK (alg_{timeframe}m BETWEEN -100 AND 100);
                """)
            
            logger.info("Created data constraints")
            
        except DatabaseError as e:
            logger.error(f"Failed to create constraints: {str(e)}")
            raise
    
    def _create_materialized_views(self) -> None:
        """Create materialized views for common queries."""
        try:
            # Create view for daily aggregates
            execute_query(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {self.table}_daily_agg AS
                SELECT 
                    DATE_TRUNC('day', timestamp) as date,
                    AVG(log_1m) as avg_log_1m,
                    AVG(alg_1m) as avg_alg_1m,
                    MAX(log_1m) as max_log_1m,
                    MIN(log_1m) as min_log_1m
                FROM {self.table}
                GROUP BY DATE_TRUNC('day', timestamp)
                WITH DATA;
            """)
            
            # Create index on the materialized view
            execute_query(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.table}_daily_agg_date
                ON {self.table}_daily_agg (date);
            """)
            
            logger.info("Created materialized views")
            
        except DatabaseError as e:
            logger.error(f"Failed to create materialized views: {str(e)}")
            raise
    
    def initialize_schema(self) -> None:
        """Initialize the complete database schema with optimizations."""
        try:
            self._create_partitioned_table()
            self._create_indexes()
            self._create_constraints()
            self._create_materialized_views()
            logger.info("Database schema initialization completed successfully")
            
        except DatabaseError as e:
            logger.error(f"Schema initialization failed: {str(e)}")
            raise
    
    def refresh_materialized_views(self) -> None:
        """Refresh materialized views with latest data."""
        try:
            execute_query(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {self.table}_daily_agg;")
            logger.info("Refreshed materialized views")
            
        except DatabaseError as e:
            logger.error(f"Failed to refresh materialized views: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        schema = DatabaseSchema()
        schema.initialize_schema()
        print("Database schema initialized successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")


# ==========================
# storage/pg_reader.py
# ==========================
# Enhanced PostgreSQL reader with connection pooling, error handling, and query optimization.

import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from db_connector import execute_query, get_metrics, DatabaseError, QueryError
from utils.logger import get_logger
import numpy as np

logger = get_logger("postgres_reader")

class PostgreSQLReader:
    """Enhanced PostgreSQL reader with optimized query execution and error handling."""
    
    def __init__(self, table: str = "signal_snapshots"):
        self.table = table
        self._validate_table()
    
    def _validate_table(self) -> None:
        """Validate that the table exists and has the correct schema."""
        try:
            query = f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
            """
            result = execute_query(query, (self.table,))
            if not result:
                raise QueryError(f"Table {self.table} does not exist")
            
            # Validate required columns
            columns = {row[0] for row in result}
            required_columns = {'timestamp'}
            if not required_columns.issubset(columns):
                raise QueryError(f"Table {self.table} is missing required columns: {required_columns - columns}")
                
        except DatabaseError as e:
            logger.error(f"Table validation failed: {str(e)}")
            raise
    
    def load_snapshots(
        self,
        days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load snapshots with optimized query and error handling.
        
        Args:
            days: Number of days to look back (ignored if start_date is provided)
            start_date: Optional start date for the query
            end_date: Optional end date for the query
            columns: Optional list of columns to retrieve
            
        Returns:
            DataFrame with the requested data
        """
        try:
            # Build the query
            select_columns = ", ".join(columns) if columns else "*"
            where_clause = []
            params = []
            
            if start_date:
                where_clause.append("timestamp >= %s")
                params.append(start_date)
            else:
                where_clause.append("timestamp >= NOW() - INTERVAL %s")
                params.append(f"{days} days")
            
            if end_date:
                where_clause.append("timestamp <= %s")
                params.append(end_date)
            
            query = f"""
                SELECT {select_columns}
                FROM {self.table}
                WHERE {' AND '.join(where_clause)}
                ORDER BY timestamp ASC
            """
            
            # Execute query and handle results
            result = execute_query(query, tuple(params))
            
            if not result:
                logger.warning(f"No data found for the specified time range in table {self.table}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=[desc[0] for desc in execute_query(f"SELECT * FROM {self.table} LIMIT 0")])
            df.set_index("timestamp", inplace=True)
            
            # Log performance metrics
            metrics = get_metrics()
            logger.info(f"Query executed in {metrics['avg_query_time']:.3f}s with {len(df)} rows")
            
            return df
            
        except DatabaseError as e:
            logger.error(f"Failed to load snapshots: {str(e)}")
            raise
    
    def get_latest_snapshot(self) -> pd.DataFrame:
        """Get the most recent snapshot with error handling."""
        try:
            query = f"""
                SELECT *
                FROM {self.table}
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = execute_query(query)
            
            if not result:
                logger.warning(f"No snapshots found in table {self.table}")
                return pd.DataFrame()
            
            df = pd.DataFrame(result, columns=[desc[0] for desc in execute_query(f"SELECT * FROM {self.table} LIMIT 0")])
            df.set_index("timestamp", inplace=True)
            return df
            
        except DatabaseError as e:
            logger.error(f"Failed to get latest snapshot: {str(e)}")
            raise
    
    def get_snapshot_count(self, days: int = 30) -> int:
        """Get the count of snapshots in the specified time range."""
        try:
            query = f"""
                SELECT COUNT(*)
                FROM {self.table}
                WHERE timestamp >= NOW() - INTERVAL %s
            """
            result = execute_query(query, (f"{days} days",))
            return result[0][0] if result else 0
            
        except DatabaseError as e:
            logger.error(f"Failed to get snapshot count: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        reader = PostgreSQLReader()
        
        # Test loading snapshots
        print("Loading snapshots from last 7 days...")
        df = reader.load_snapshots(days=7)
        print(f"Loaded {len(df)} snapshots")
        print(df.head())
        
        # Test getting latest snapshot
        print("\nGetting latest snapshot...")
        latest = reader.get_latest_snapshot()
        print(latest)
        
        # Test getting snapshot count
        print("\nGetting snapshot count...")
        count = reader.get_snapshot_count(days=7)
        print(f"Total snapshots in last 7 days: {count}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

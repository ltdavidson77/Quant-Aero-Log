# ==========================
# storage/pg_writer.py
# ==========================
# Enhanced PostgreSQL writer with connection pooling, error handling, and batch writing.

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
from db_connector import execute_query, get_metrics, DatabaseError, QueryError
from utils.logger import get_logger
import numpy as np

logger = get_logger("postgres_writer")

class PostgreSQLWriter:
    """Enhanced PostgreSQL writer with optimized batch writing and error handling."""
    
    def __init__(self, table: str = "signal_snapshots", batch_size: int = 1000):
        self.table = table
        self.batch_size = batch_size
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
    
    def write_snapshot(self, df: pd.DataFrame) -> None:
        """
        Write a single snapshot to the database with error handling.
        
        Args:
            df: DataFrame containing the snapshot data
        """
        try:
            if df.empty:
                logger.warning("Attempted to write empty DataFrame")
                return
            
            # Prepare data
            df = df.copy()
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.index
            
            # Get column names
            columns = df.columns.tolist()
            
            # Build the query
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"""
                INSERT INTO {self.table} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (timestamp) DO UPDATE SET
                    {", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != 'timestamp')}
            """
            
            # Execute query
            values = df.iloc[0].tolist()
            execute_query(query, tuple(values))
            
            # Log performance metrics
            metrics = get_metrics()
            logger.info(f"Snapshot written in {metrics['avg_query_time']:.3f}s")
            
        except DatabaseError as e:
            logger.error(f"Failed to write snapshot: {str(e)}")
            raise
    
    def write_snapshots(self, df: pd.DataFrame) -> None:
        """
        Write multiple snapshots in batches with error handling.
        
        Args:
            df: DataFrame containing multiple snapshots
        """
        try:
            if df.empty:
                logger.warning("Attempted to write empty DataFrame")
                return
            
            # Prepare data
            df = df.copy()
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.index
            
            # Get column names
            columns = df.columns.tolist()
            
            # Build the query
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"""
                INSERT INTO {self.table} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (timestamp) DO UPDATE SET
                    {", ".join(f"{col} = EXCLUDED.{col}" for col in columns if col != 'timestamp')}
            """
            
            # Process in batches
            total_rows = len(df)
            for i in range(0, total_rows, self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                values = [tuple(row) for row in batch.values]
                
                # Execute batch query
                execute_query(query, values)
                
                # Log progress
                progress = min(i + self.batch_size, total_rows)
                logger.info(f"Written {progress}/{total_rows} snapshots")
            
            # Log final performance metrics
            metrics = get_metrics()
            logger.info(f"All snapshots written in {metrics['avg_query_time']:.3f}s per batch")
            
        except DatabaseError as e:
            logger.error(f"Failed to write snapshots: {str(e)}")
            raise
    
    def delete_snapshots(self, start_date: datetime, end_date: Optional[datetime] = None) -> int:
        """
        Delete snapshots within a date range with error handling.
        
        Args:
            start_date: Start date for deletion
            end_date: Optional end date for deletion
            
        Returns:
            Number of rows deleted
        """
        try:
            # Build the query
            where_clause = ["timestamp >= %s"]
            params = [start_date]
            
            if end_date:
                where_clause.append("timestamp <= %s")
                params.append(end_date)
            
            query = f"""
                DELETE FROM {self.table}
                WHERE {' AND '.join(where_clause)}
                RETURNING timestamp
            """
            
            # Execute query
            result = execute_query(query, tuple(params))
            deleted_count = len(result)
            
            logger.info(f"Deleted {deleted_count} snapshots")
            return deleted_count
            
        except DatabaseError as e:
            logger.error(f"Failed to delete snapshots: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        from generate_data import get_price_series
        from multi_timeframe import compute_multi_timeframe_signals
        
        # Initialize writer
        writer = PostgreSQLWriter()
        
        # Generate test data
        print("Generating test data...")
        df = get_price_series()
        signals = compute_multi_timeframe_signals(df)
        
        # Test writing single snapshot
        print("\nWriting single snapshot...")
        writer.write_snapshot(signals.iloc[[-1]])
        
        # Test writing multiple snapshots
        print("\nWriting multiple snapshots...")
        writer.write_snapshots(signals.tail(10))
        
        # Test deleting snapshots
        print("\nDeleting snapshots...")
        start_date = signals.index[-10]
        deleted = writer.delete_snapshots(start_date)
        print(f"Deleted {deleted} snapshots")
        
    except Exception as e:
        print(f"Error: {str(e)}")

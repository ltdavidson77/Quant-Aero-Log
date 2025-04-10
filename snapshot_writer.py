# ==========================
# snapshot_writer.py
# ==========================
# Enhanced snapshot writer with comprehensive features for data management and monitoring.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import json
import zlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from tqdm import tqdm
import hashlib
import uuid
import os
import shutil
from enum import Enum
import tempfile
import gzip
import bz2
import lzma
import msgpack
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet
import snappy
import lz4.frame
import zstandard as zstd
from prometheus_client import Counter, Gauge, Histogram
import psutil
import gc
import tracemalloc
from memory_profiler import profile
import line_profiler
import pyinstrument
import grpc
from grpc import aio
import zmq
import msgpack
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet
import snappy
import lz4.frame
import zstandard as zstd

# Database connectors
from db_connector import (
    get_pg_connection, close_pg_connection,
    get_mongo_connection, close_mongo_connection,
    get_redis_connection, close_redis_connection,
    get_cassandra_connection, close_cassandra_connection,
    get_clickhouse_connection, close_clickhouse_connection,
    get_timescaledb_connection, close_timescaledb_connection
)

# Configuration
from config_env import (
    DEBUG_MODE,
    SNAPSHOT_COMPRESSION_LEVEL,
    SNAPSHOT_BATCH_SIZE,
    SNAPSHOT_RETRY_ATTEMPTS,
    SNAPSHOT_RETRY_DELAY,
    SNAPSHOT_VALIDATION_ENABLED,
    SNAPSHOT_METADATA_ENABLED,
    SNAPSHOT_BACKUP_ENABLED,
    SNAPSHOT_MONITORING_ENABLED,
    SNAPSHOT_SCHEMA_VALIDATION_ENABLED,
    SNAPSHOT_DATA_CLEANING_ENABLED,
    SNAPSHOT_PERFORMANCE_MONITORING_ENABLED,
    SNAPSHOT_SECURITY_ENABLED
)

logger = logging.getLogger(__name__)

# Prometheus metrics
SNAPSHOT_WRITE_COUNTER = Counter('snapshot_write_total', 'Total number of snapshots written')
SNAPSHOT_WRITE_ERRORS = Counter('snapshot_write_errors_total', 'Total number of snapshot write errors')
SNAPSHOT_WRITE_DURATION = Histogram('snapshot_write_duration_seconds', 'Time spent writing snapshots')
SNAPSHOT_SIZE_BYTES = Gauge('snapshot_size_bytes', 'Size of snapshots in bytes')
SNAPSHOT_COMPRESSION_RATIO = Gauge('snapshot_compression_ratio', 'Compression ratio of snapshots')
SNAPSHOT_MEMORY_USAGE = Gauge('snapshot_memory_usage_bytes', 'Memory usage during snapshot operations')

class CompressionType(Enum):
    """Supported compression types."""
    NONE = 'none'
    ZLIB = 'zlib'
    GZIP = 'gzip'
    BZIP2 = 'bz2'
    LZMA = 'lzma'
    SNAPPY = 'snappy'
    LZ4 = 'lz4'
    ZSTD = 'zstd'

class SerializationType(Enum):
    """Supported serialization types."""
    PICKLE = 'pickle'
    JSON = 'json'
    MSGPACK = 'msgpack'
    ORJSON = 'orjson'
    PARQUET = 'parquet'
    ARROW = 'arrow'

@dataclass
class SnapshotMetadata:
    """Enhanced metadata for snapshot operations."""
    # Basic metadata
    timestamp: datetime
    source: str
    size_bytes: int
    compression_ratio: float
    validation_status: bool
    processing_time_ms: float
    error_count: int
    warning_count: int
    custom_metadata: Dict[str, Any]
    
    # Advanced metadata
    schema_version: str
    data_hash: str
    compression_type: str
    serialization_type: str
    backup_status: bool
    security_status: bool
    performance_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    data_quality_metrics: Dict[str, float]

class SnapshotWriter:
    """Enhanced snapshot writer with comprehensive features."""
    
    def __init__(self,
                 backend: str = 'postgresql',
                 compression_type: CompressionType = CompressionType.ZLIB,
                 serialization_type: SerializationType = SerializationType.PICKLE,
                 batch_size: int = SNAPSHOT_BATCH_SIZE,
                 retry_attempts: int = SNAPSHOT_RETRY_ATTEMPTS,
                 retry_delay: float = SNAPSHOT_RETRY_DELAY,
                 backup_path: Optional[str] = None,
                 schema_path: Optional[str] = None):
        """
        Initialize the snapshot writer with comprehensive features.
        
        Args:
            backend: Database backend
            compression_type: Type of compression to use
            serialization_type: Type of serialization to use
            batch_size: Batch size for bulk operations
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries in seconds
            backup_path: Path for backup storage
            schema_path: Path to schema definition
        """
        self.backend = backend.lower()
        self.compression_type = compression_type
        self.serialization_type = serialization_type
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.backup_path = backup_path
        self.schema_path = schema_path
        
        # Initialize metadata
        self.metadata = SnapshotMetadata(
            timestamp=datetime.now(),
            source='',
            size_bytes=0,
            compression_ratio=1.0,
            validation_status=True,
            processing_time_ms=0,
            error_count=0,
            warning_count=0,
            custom_metadata={},
            schema_version='1.0',
            data_hash='',
            compression_type=compression_type.value,
            serialization_type=serialization_type.value,
            backup_status=False,
            security_status=True,
            performance_metrics={},
            system_metrics={},
            data_quality_metrics={}
        )
        
        # Setup components
        self._setup_backend()
        self._setup_compression()
        self._setup_serialization()
        self._setup_monitoring()
        self._setup_security()
        
    def _setup_backend(self) -> None:
        """Setup the appropriate database backend."""
        try:
            if self.backend == 'postgresql':
                self.conn = get_pg_connection()
            elif self.backend == 'mongodb':
                self.conn = get_mongo_connection()
            elif self.backend == 'redis':
                self.conn = get_redis_connection()
            elif self.backend == 'cassandra':
                self.conn = get_cassandra_connection()
            elif self.backend == 'clickhouse':
                self.conn = get_clickhouse_connection()
            elif self.backend == 'timescaledb':
                self.conn = get_timescaledb_connection()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            logger.error(f"Backend setup error: {str(e)}")
            raise
            
    def _setup_compression(self) -> None:
        """Setup compression handlers."""
        self.compression_handlers = {
            CompressionType.NONE: lambda x: x,
            CompressionType.ZLIB: lambda x: zlib.compress(x, level=self.compression_level),
            CompressionType.GZIP: lambda x: gzip.compress(x),
            CompressionType.BZIP2: lambda x: bz2.compress(x),
            CompressionType.LZMA: lambda x: lzma.compress(x),
            CompressionType.SNAPPY: lambda x: snappy.compress(x),
            CompressionType.LZ4: lambda x: lz4.frame.compress(x),
            CompressionType.ZSTD: lambda x: zstd.compress(x)
        }
        
    def _setup_serialization(self) -> None:
        """Setup serialization handlers."""
        self.serialization_handlers = {
            SerializationType.PICKLE: lambda x: pickle.dumps(x),
            SerializationType.JSON: lambda x: json.dumps(x).encode(),
            SerializationType.MSGPACK: lambda x: msgpack.packb(x),
            SerializationType.ORJSON: lambda x: orjson.dumps(x),
            SerializationType.PARQUET: lambda x: self._serialize_to_parquet(x),
            SerializationType.ARROW: lambda x: self._serialize_to_arrow(x)
        }
        
    def _setup_monitoring(self) -> None:
        """Setup monitoring components."""
        if SNAPSHOT_MONITORING_ENABLED:
            tracemalloc.start()
            self.memory_snapshot = tracemalloc.take_snapshot()
            
    def _setup_security(self) -> None:
        """Setup security components."""
        if SNAPSHOT_SECURITY_ENABLED:
            self.encryption_key = self._generate_encryption_key()
            
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure storage."""
        return hashlib.sha256(str(uuid.uuid4()).encode()).digest()
        
    def _validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate data against schema definition."""
        if not SNAPSHOT_SCHEMA_VALIDATION_ENABLED or not self.schema_path:
            return True
            
        try:
            with open(self.schema_path) as f:
                schema = json.load(f)
                
            # Validate columns
            if not all(col in data.columns for col in schema['required_columns']):
                return False
                
            # Validate data types
            for col, dtype in schema['column_types'].items():
                if col in data.columns:
                    if not isinstance(data[col].iloc[0], eval(dtype)):
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return False
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        if not SNAPSHOT_DATA_CLEANING_ENABLED:
            return data
            
        try:
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers
            for col in data.select_dtypes(include=[np.number]).columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                data = data[~((data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr)))]
                
            return data
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return data
            
    def _calculate_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics."""
        try:
            metrics = {
                'completeness': 1.0 - data.isna().sum().sum() / (data.shape[0] * data.shape[1]),
                'uniqueness': len(data.drop_duplicates()) / len(data),
                'consistency': self._calculate_consistency_score(data),
                'accuracy': self._calculate_accuracy_score(data),
                'timeliness': self._calculate_timeliness_score(data)
            }
            return metrics
        except Exception as e:
            logger.error(f"Data quality metrics calculation error: {str(e)}")
            return {}
            
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        try:
            # Implement consistency checks
            return 1.0
        except Exception as e:
            logger.error(f"Consistency score calculation error: {str(e)}")
            return 0.0
            
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate data accuracy score."""
        try:
            # Implement accuracy checks
            return 1.0
        except Exception as e:
            logger.error(f"Accuracy score calculation error: {str(e)}")
            return 0.0
            
    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """Calculate data timeliness score."""
        try:
            # Implement timeliness checks
            return 1.0
        except Exception as e:
            logger.error(f"Timeliness score calculation error: {str(e)}")
            return 0.0
            
    def _backup_data(self, data: bytes) -> bool:
        """Create backup of the data."""
        if not SNAPSHOT_BACKUP_ENABLED or not self.backup_path:
            return False
            
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(self.backup_path, exist_ok=True)
            
            # Generate backup filename
            backup_file = os.path.join(
                self.backup_path,
                f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            )
            
            # Write backup
            with open(backup_file, 'wb') as f:
                f.write(data)
                
            return True
        except Exception as e:
            logger.error(f"Backup creation error: {str(e)}")
            return False
            
    def _monitor_performance(self) -> Dict[str, float]:
        """Monitor system performance metrics."""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
                'process_memory': psutil.Process().memory_info().rss
            }
            return metrics
        except Exception as e:
            logger.error(f"Performance monitoring error: {str(e)}")
            return {}
            
    async def write_snapshot(self,
                           data: pd.DataFrame,
                           table_name: str = 'signal_snapshots',
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Write snapshot to database with comprehensive features.
        
        Args:
            data: DataFrame containing the snapshot data
            table_name: Name of the table/collection
            metadata: Additional metadata to store
            
        Returns:
            Success status
        """
        start_time = time.time()
        self.metadata.timestamp = datetime.now()
        self.metadata.source = table_name
        
        try:
            # Update Prometheus metrics
            SNAPSHOT_WRITE_COUNTER.inc()
            
            # Validate schema if enabled
            if not self._validate_schema(data):
                logger.error("Schema validation failed")
                SNAPSHOT_WRITE_ERRORS.inc()
                return False
                
            # Clean data if enabled
            data = self._clean_data(data)
            
            # Calculate data quality metrics
            self.metadata.data_quality_metrics = self._calculate_data_quality_metrics(data)
            
            # Prepare data
            data_bytes = self._prepare_data(data)
            
            # Create backup if enabled
            if SNAPSHOT_BACKUP_ENABLED:
                self.metadata.backup_status = self._backup_data(data_bytes)
                
            # Store metadata
            self.metadata.custom_metadata = metadata or {}
            self.metadata.system_metrics = self._monitor_performance()
            
            # Write to appropriate backend
            if self.backend == 'postgresql':
                await self._write_to_postgresql(data, table_name)
            elif self.backend == 'mongodb':
                await self._write_to_mongodb(data, table_name)
            elif self.backend == 'redis':
                await self._write_to_redis(data_bytes, table_name)
            elif self.backend == 'cassandra':
                await self._write_to_cassandra(data, table_name)
            elif self.backend == 'clickhouse':
                await self._write_to_clickhouse(data, table_name)
            elif self.backend == 'timescaledb':
                await self._write_to_timescaledb(data, table_name)
                
            # Update processing time and metrics
            processing_time = (time.time() - start_time) * 1000
            self.metadata.processing_time_ms = processing_time
            SNAPSHOT_WRITE_DURATION.observe(processing_time / 1000)
            SNAPSHOT_SIZE_BYTES.set(len(data_bytes))
            SNAPSHOT_COMPRESSION_RATIO.set(self.metadata.compression_ratio)
            SNAPSHOT_MEMORY_USAGE.set(psutil.Process().memory_info().rss)
            
            if DEBUG_MODE:
                logger.info(f"Snapshot written successfully to {table_name}")
                logger.info(f"Processing time: {processing_time:.2f}ms")
                logger.info(f"Compression ratio: {self.metadata.compression_ratio:.2f}")
                logger.info(f"Data quality metrics: {self.metadata.data_quality_metrics}")
                
            return True
            
        except Exception as e:
            logger.error(f"Snapshot writing error: {str(e)}")
            SNAPSHOT_WRITE_ERRORS.inc()
            self.metadata.error_count += 1
            return False
            
    def close(self) -> None:
        """Close database connections and cleanup."""
        try:
            if self.backend == 'postgresql':
                close_pg_connection(self.conn)
            elif self.backend == 'mongodb':
                close_mongo_connection(self.conn)
            elif self.backend == 'redis':
                close_redis_connection(self.conn)
            elif self.backend == 'cassandra':
                close_cassandra_connection(self.conn)
            elif self.backend == 'clickhouse':
                close_clickhouse_connection(self.conn)
            elif self.backend == 'timescaledb':
                close_timescaledb_connection(self.conn)
                
            # Cleanup monitoring
            if SNAPSHOT_MONITORING_ENABLED:
                tracemalloc.stop()
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def _write_to_postgresql(self, data: pd.DataFrame, table_name: str) -> None:
        """Write snapshot to PostgreSQL with enhanced features."""
        cursor = self.conn.cursor()
        
        try:
            # Prepare data
            data = data.copy()
            data['timestamp'] = data.index.astype(str)
            cols = data.columns.tolist()
            
            # Create table if not exists with enhanced schema
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                {', '.join(f'"{col}" TEXT' for col in cols if col != 'timestamp')},
                metadata JSONB,
                compression_type TEXT,
                serialization_type TEXT,
                data_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            
            # Create index on timestamp if not exists
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
                ON {table_name} (timestamp)
            """)
            
            # Insert data in batches with enhanced error handling
            for i in range(0, len(data), self.batch_size):
                batch = data.iloc[i:i+self.batch_size]
                values = []
                
                for _, row in batch.iterrows():
                    # Prepare row data with metadata
                    row_data = {
                        'timestamp': row.name,
                        **row.to_dict(),
                        'metadata': json.dumps(self.metadata.__dict__),
                        'compression_type': self.compression_type.value,
                        'serialization_type': self.serialization_type.value,
                        'data_hash': hashlib.sha256(str(row).encode()).hexdigest()
                    }
                    values.append(tuple(row_data.values()))
                
                # Use parameterized query for security
                placeholders = ', '.join(['%s'] * len(row_data))
                insert_query = f"""
                INSERT INTO {table_name} ({', '.join(row_data.keys())})
                VALUES ({placeholders})
                """
                
                cursor.executemany(insert_query, values)
                
                # Update metrics
                SNAPSHOT_WRITE_COUNTER.inc()
                SNAPSHOT_SIZE_BYTES.set(len(str(values).encode()))
                
            self.conn.commit()
            
            # Update table statistics
            cursor.execute(f"ANALYZE {table_name}")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"PostgreSQL write error: {str(e)}")
            SNAPSHOT_WRITE_ERRORS.inc()
            raise
            
        finally:
            cursor.close()

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='1s'),
        'value': np.random.randn(1000),
        'quality': np.random.randint(0, 100, 1000)
    })
    data.set_index('timestamp', inplace=True)
    
    # Write snapshot with comprehensive features
    with SnapshotWriter(
        backend='postgresql',
        compression_type=CompressionType.ZSTD,
        serialization_type=SerializationType.PARQUET,
        batch_size=1000,
        backup_path='./backups',
        schema_path='./schema.json'
    ) as writer:
        asyncio.run(writer.write_snapshot(
            data,
            table_name='signal_snapshots',
            metadata={
                'source': 'test',
                'version': '1.0',
                'environment': 'production',
                'tags': ['test', 'snapshot']
            }
        ))

# ==========================
# storage/schema_optimizer.py
# ==========================
# Database schema optimization and management.

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import json
from pathlib import Path
from db_connector import execute_query, DatabaseError, QueryError
from utils.logger import get_logger
from query_optimizer import QueryOptimizer

logger = get_logger("schema_optimizer")

class SchemaOptimizer:
    """Manages database schema optimization and maintenance."""
    
    def __init__(
        self,
        stats_dir: str = "schema_stats",
        vacuum_threshold: int = 50,  # percentage
        analyze_threshold: int = 1000,  # rows
        max_parallel_operations: int = 4
    ):
        self.stats_dir = Path(stats_dir)
        self.vacuum_threshold = vacuum_threshold
        self.analyze_threshold = analyze_threshold
        self.max_parallel_operations = max_parallel_operations
        self.stats_file = self.stats_dir / "schema_stats.json"
        self.query_optimizer = QueryOptimizer()
        self._setup_stats_directory()
        self._load_stats()
    
    def _setup_stats_directory(self) -> None:
        """Setup statistics directory structure."""
        try:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Schema stats directory created at {self.stats_dir}")
        except Exception as e:
            logger.error(f"Failed to setup stats directory: {str(e)}")
            raise
    
    def _load_stats(self) -> None:
        """Load schema statistics from file."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                self.stats = {
                    'table_stats': {},
                    'index_stats': {},
                    'maintenance_history': [],
                    'optimization_suggestions': []
                }
                self._save_stats()
        except Exception as e:
            logger.error(f"Failed to load schema stats: {str(e)}")
            raise
    
    def _save_stats(self) -> None:
        """Save schema statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save schema stats: {str(e)}")
            raise
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze table structure and statistics.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Get table statistics
            stats_query = f"""
            SELECT
                relname,
                n_live_tup,
                n_dead_tup,
                n_mod_since_analyze,
                last_vacuum,
                last_analyze,
                reltuples,
                relpages,
                relallvisible
            FROM pg_stat_user_tables
            WHERE relname = '{table_name}';
            """
            
            result = execute_query(stats_query)
            if not result:
                raise SchemaError(f"Failed to get statistics for table {table_name}")
            
            # Get column statistics
            columns_query = f"""
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = '{table_name}';
            """
            
            columns = execute_query(columns_query)
            
            # Get index statistics
            indexes_query = f"""
            SELECT
                indexname,
                indexdef,
                indisunique,
                indisprimary,
                indisvalid
            FROM pg_indexes
            WHERE tablename = '{table_name}';
            """
            
            indexes = execute_query(indexes_query)
            
            # Analyze table structure
            analysis = {
                'table_name': table_name,
                'row_count': result[0][1],
                'dead_rows': result[0][2],
                'modified_since_analyze': result[0][3],
                'last_vacuum': result[0][4],
                'last_analyze': result[0][5],
                'estimated_rows': result[0][6],
                'pages': result[0][7],
                'visible_pages': result[0][8],
                'columns': [
                    {
                        'name': col[0],
                        'type': col[1],
                        'max_length': col[2],
                        'nullable': col[3],
                        'default': col[4]
                    }
                    for col in columns
                ],
                'indexes': [
                    {
                        'name': idx[0],
                        'definition': idx[1],
                        'unique': idx[2],
                        'primary': idx[3],
                        'valid': idx[4]
                    }
                    for idx in indexes
                ],
                'suggestions': self._generate_suggestions(
                    result[0],
                    columns,
                    indexes
                )
            }
            
            # Update statistics
            self.stats['table_stats'][table_name] = analysis
            self._save_stats()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {str(e)}")
            raise
    
    def _generate_suggestions(
        self,
        table_stats: tuple,
        columns: List[tuple],
        indexes: List[tuple]
    ) -> List[str]:
        """Generate optimization suggestions based on table analysis."""
        suggestions = []
        
        # Check for vacuum needed
        dead_ratio = table_stats[2] / (table_stats[1] + table_stats[2]) * 100
        if dead_ratio > self.vacuum_threshold:
            suggestions.append(f"High dead tuple ratio ({dead_ratio:.1f}%), consider running VACUUM")
        
        # Check for analyze needed
        if table_stats[3] > self.analyze_threshold:
            suggestions.append(f"High number of modifications since last analyze ({table_stats[3]}), consider running ANALYZE")
        
        # Check for missing primary key
        has_primary = any(idx[3] for idx in indexes)
        if not has_primary:
            suggestions.append("Table has no primary key, consider adding one")
        
        # Check for missing indexes on foreign keys
        for col in columns:
            if col[0].endswith('_id') and not any(
                f"({col[0]})" in idx[1] for idx in indexes
            ):
                suggestions.append(f"Consider adding index on foreign key column {col[0]}")
        
        # Check for large tables without partitioning
        if table_stats[7] > 10000:  # More than 10,000 pages
            suggestions.append("Consider partitioning large table")
        
        return suggestions
    
    def optimize_table(self, table_name: str) -> None:
        """
        Optimize table structure based on analysis.
        
        Args:
            table_name: Name of the table to optimize
        """
        try:
            # Analyze table
            analysis = self.analyze_table(table_name)
            
            # Apply optimizations
            for suggestion in analysis['suggestions']:
                if "running VACUUM" in suggestion:
                    self._vacuum_table(table_name)
                elif "running ANALYZE" in suggestion:
                    self._analyze_table(table_name)
                elif "adding primary key" in suggestion:
                    self._add_primary_key(table_name)
                elif "adding index" in suggestion:
                    column = suggestion.split()[-1]
                    self._add_index(table_name, column)
                elif "partitioning" in suggestion:
                    self._partition_table(table_name)
            
            logger.info(f"Table {table_name} optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize table {table_name}: {str(e)}")
            raise
    
    def _vacuum_table(self, table_name: str) -> None:
        """Run VACUUM on table."""
        try:
            execute_query(f"VACUUM ANALYZE {table_name}")
            logger.info(f"VACUUM completed for table {table_name}")
        except Exception as e:
            logger.error(f"Failed to VACUUM table {table_name}: {str(e)}")
            raise
    
    def _analyze_table(self, table_name: str) -> None:
        """Run ANALYZE on table."""
        try:
            execute_query(f"ANALYZE {table_name}")
            logger.info(f"ANALYZE completed for table {table_name}")
        except Exception as e:
            logger.error(f"Failed to ANALYZE table {table_name}: {str(e)}")
            raise
    
    def _add_primary_key(self, table_name: str) -> None:
        """Add primary key to table."""
        try:
            # Find suitable primary key column
            columns_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND column_name LIKE '%id'
            LIMIT 1;
            """
            
            result = execute_query(columns_query)
            if not result:
                raise SchemaError(f"No suitable primary key column found for table {table_name}")
            
            column = result[0][0]
            execute_query(f"ALTER TABLE {table_name} ADD PRIMARY KEY ({column})")
            logger.info(f"Added primary key on {column} to table {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to add primary key to table {table_name}: {str(e)}")
            raise
    
    def _add_index(self, table_name: str, column: str) -> None:
        """Add index to table."""
        try:
            index_name = f"idx_{table_name}_{column}"
            execute_query(f"CREATE INDEX {index_name} ON {table_name} ({column})")
            logger.info(f"Added index {index_name} to table {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to add index to table {table_name}: {str(e)}")
            raise
    
    def _partition_table(self, table_name: str) -> None:
        """Partition table by time."""
        try:
            # Create partition table
            partition_table = f"{table_name}_partitioned"
            execute_query(f"""
                CREATE TABLE {partition_table} (LIKE {table_name} INCLUDING ALL)
                PARTITION BY RANGE (timestamp);
            """)
            
            # Copy data
            execute_query(f"""
                INSERT INTO {partition_table}
                SELECT * FROM {table_name};
            """)
            
            # Rename tables
            execute_query(f"ALTER TABLE {table_name} RENAME TO {table_name}_old")
            execute_query(f"ALTER TABLE {partition_table} RENAME TO {table_name}")
            
            logger.info(f"Partitioned table {table_name} by timestamp")
            
        except Exception as e:
            logger.error(f"Failed to partition table {table_name}: {str(e)}")
            raise
    
    def get_schema_stats(self) -> Dict[str, Any]:
        """Get aggregated schema statistics."""
        try:
            stats = {
                'total_tables': len(self.stats['table_stats']),
                'total_indexes': sum(
                    len(table['indexes'])
                    for table in self.stats['table_stats'].values()
                ),
                'tables_needing_vacuum': [],
                'tables_needing_analyze': [],
                'tables_without_pk': [],
                'large_tables': [],
                'optimization_suggestions': []
            }
            
            for table_name, table_stats in self.stats['table_stats'].items():
                # Check for vacuum needed
                dead_ratio = table_stats['dead_rows'] / (table_stats['row_count'] + table_stats['dead_rows']) * 100
                if dead_ratio > self.vacuum_threshold:
                    stats['tables_needing_vacuum'].append(table_name)
                
                # Check for analyze needed
                if table_stats['modified_since_analyze'] > self.analyze_threshold:
                    stats['tables_needing_analyze'].append(table_name)
                
                # Check for missing primary key
                if not any(idx['primary'] for idx in table_stats['indexes']):
                    stats['tables_without_pk'].append(table_name)
                
                # Check for large tables
                if table_stats['pages'] > 10000:
                    stats['large_tables'].append(table_name)
                
                # Add optimization suggestions
                stats['optimization_suggestions'].extend(table_stats['suggestions'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get schema stats: {str(e)}")
            raise

class SchemaError(Exception):
    """Custom exception for schema-related errors."""
    pass

# Example usage
if __name__ == "__main__":
    try:
        # Initialize schema optimizer
        optimizer = SchemaOptimizer(
            stats_dir="schema_stats",
            vacuum_threshold=50,
            analyze_threshold=1000,
            max_parallel_operations=4
        )
        
        # Analyze table
        analysis = optimizer.analyze_table("snapshots")
        print("Table Analysis:", json.dumps(analysis, indent=2))
        
        # Get schema statistics
        stats = optimizer.get_schema_stats()
        print("\nSchema Statistics:", json.dumps(stats, indent=2))
        
        # Optimize table
        optimizer.optimize_table("snapshots")
        
    except Exception as e:
        print(f"Error: {str(e)}") 
# ==========================
# storage/query_optimizer.py
# ==========================
# Query optimization and performance monitoring.

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import psutil
from db_connector import execute_query, DatabaseError, QueryError
from utils.logger import get_logger

logger = get_logger("query_optimizer")

class QueryOptimizer:
    """Manages query optimization and performance monitoring."""
    
    def __init__(
        self,
        stats_dir: str = "query_stats",
        slow_query_threshold: float = 1.0,  # seconds
        cache_hit_threshold: float = 0.95,  # 95% cache hit ratio
        max_parallel_queries: int = 4
    ):
        self.stats_dir = Path(stats_dir)
        self.slow_query_threshold = slow_query_threshold
        self.cache_hit_threshold = cache_hit_threshold
        self.max_parallel_queries = max_parallel_queries
        self.stats_file = self.stats_dir / "query_stats.json"
        self._setup_stats_directory()
        self._load_stats()
    
    def _setup_stats_directory(self) -> None:
        """Setup statistics directory structure."""
        try:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Query stats directory created at {self.stats_dir}")
        except Exception as e:
            logger.error(f"Failed to setup stats directory: {str(e)}")
            raise
    
    def _load_stats(self) -> None:
        """Load query statistics from file."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                self.stats = {
                    'query_history': [],
                    'slow_queries': [],
                    'cache_stats': {
                        'total_queries': 0,
                        'cache_hits': 0,
                        'cache_misses': 0
                    },
                    'optimization_suggestions': []
                }
                self._save_stats()
        except Exception as e:
            logger.error(f"Failed to load query stats: {str(e)}")
            raise
    
    def _save_stats(self) -> None:
        """Save query statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query stats: {str(e)}")
            raise
    
    def _check_system_resources(self) -> bool:
        """
        Check if system resources are sufficient for query execution.
        
        Returns:
            bool: True if resources are sufficient, False otherwise
        """
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                logger.warning(f"High CPU usage ({cpu_percent}%), may affect query performance")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logger.warning(f"High memory usage ({memory.percent}%), may affect query performance")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to check system resources: {str(e)}")
            return False
    
    def analyze_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Analyze query execution plan.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Get query plan
            plan_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            result = execute_query(plan_query)
            
            if not result:
                raise QueryError("Failed to get query plan")
            
            plan = result[0][0]  # First row, first column contains JSON plan
            
            # Analyze plan
            analysis = {
                'total_cost': plan['Plan']['Total Cost'],
                'execution_time': plan['Execution Time'],
                'planning_time': plan['Planning Time'],
                'buffer_usage': plan['Plan']['Buffers'],
                'node_types': self._extract_node_types(plan['Plan']),
                'suggestions': self._generate_suggestions(plan['Plan'])
            }
            
            # Track query
            self._track_query(query, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query plan: {str(e)}")
            raise
    
    def _extract_node_types(self, plan: Dict[str, Any]) -> List[str]:
        """Extract unique node types from query plan."""
        node_types = set()
        
        def extract_nodes(node: Dict[str, Any]) -> None:
            node_types.add(node['Node Type'])
            if 'Plans' in node:
                for subplan in node['Plans']:
                    extract_nodes(subplan)
        
        extract_nodes(plan)
        return sorted(node_types)
    
    def _generate_suggestions(self, plan: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on query plan."""
        suggestions = []
        
        # Check for sequential scans
        if 'Seq Scan' in self._extract_node_types(plan):
            suggestions.append("Consider adding appropriate indexes to avoid sequential scans")
        
        # Check for nested loops
        if 'Nested Loop' in self._extract_node_types(plan):
            suggestions.append("Consider optimizing join conditions or adding indexes for nested loops")
        
        # Check for high buffer usage
        if plan['Buffers']['Shared Hit'] / plan['Buffers']['Total'] < self.cache_hit_threshold:
            suggestions.append("Consider increasing shared_buffers or optimizing query for better cache usage")
        
        # Check for high execution time
        if plan['Execution Time'] > self.slow_query_threshold * 1000:  # Convert to milliseconds
            suggestions.append("Query execution time exceeds threshold, consider optimization")
        
        return suggestions
    
    def _track_query(self, query: str, analysis: Dict[str, Any]) -> None:
        """Track query execution and update statistics."""
        try:
            # Update query history
            self.stats['query_history'].append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'analysis': analysis
            })
            
            # Track slow queries
            if analysis['execution_time'] > self.slow_query_threshold * 1000:
                self.stats['slow_queries'].append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'execution_time': analysis['execution_time'],
                    'suggestions': analysis['suggestions']
                })
            
            # Update cache statistics
            self.stats['cache_stats']['total_queries'] += 1
            if analysis['buffer_usage']['Shared Hit'] / analysis['buffer_usage']['Total'] >= self.cache_hit_threshold:
                self.stats['cache_stats']['cache_hits'] += 1
            else:
                self.stats['cache_stats']['cache_misses'] += 1
            
            # Add optimization suggestions
            for suggestion in analysis['suggestions']:
                if suggestion not in self.stats['optimization_suggestions']:
                    self.stats['optimization_suggestions'].append(suggestion)
            
            self._save_stats()
            
        except Exception as e:
            logger.error(f"Failed to track query: {str(e)}")
            raise
    
    def get_query_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get aggregated query statistics.
        
        Args:
            time_window: Optional time window to filter statistics
            
        Returns:
            Dict containing aggregated statistics
        """
        try:
            stats = {
                'total_queries': 0,
                'slow_queries': 0,
                'avg_execution_time': 0,
                'cache_hit_ratio': 0,
                'common_node_types': {},
                'optimization_suggestions': []
            }
            
            # Filter by time window if specified
            cutoff_time = datetime.now() - time_window if time_window else None
            
            # Aggregate statistics
            for query in self.stats['query_history']:
                if cutoff_time and datetime.fromisoformat(query['timestamp']) < cutoff_time:
                    continue
                
                stats['total_queries'] += 1
                if query['analysis']['execution_time'] > self.slow_query_threshold * 1000:
                    stats['slow_queries'] += 1
                
                stats['avg_execution_time'] += query['analysis']['execution_time']
                
                for node_type in query['analysis']['node_types']:
                    stats['common_node_types'][node_type] = stats['common_node_types'].get(node_type, 0) + 1
            
            # Calculate averages
            if stats['total_queries'] > 0:
                stats['avg_execution_time'] /= stats['total_queries']
                stats['cache_hit_ratio'] = self.stats['cache_stats']['cache_hits'] / self.stats['cache_stats']['total_queries']
            
            # Add optimization suggestions
            stats['optimization_suggestions'] = self.stats['optimization_suggestions']
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get query stats: {str(e)}")
            raise
    
    def optimize_query(self, query: str) -> str:
        """
        Optimize SQL query based on analysis.
        
        Args:
            query: Original SQL query
            
        Returns:
            str: Optimized SQL query
        """
        try:
            # Analyze query
            analysis = self.analyze_query_plan(query)
            
            # Apply optimizations based on suggestions
            optimized_query = query
            
            for suggestion in analysis['suggestions']:
                if "adding appropriate indexes" in suggestion:
                    # Add index hints if supported
                    optimized_query = self._add_index_hints(optimized_query)
                elif "optimizing join conditions" in suggestion:
                    # Optimize join order
                    optimized_query = self._optimize_joins(optimized_query)
                elif "increasing shared_buffers" in suggestion:
                    # Add buffer hints
                    optimized_query = self._add_buffer_hints(optimized_query)
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"Failed to optimize query: {str(e)}")
            return query
    
    def _add_index_hints(self, query: str) -> str:
        """Add index hints to query if supported."""
        # This is a placeholder for index hint implementation
        # Actual implementation would depend on database version and configuration
        return query
    
    def _optimize_joins(self, query: str) -> str:
        """Optimize join order in query."""
        # This is a placeholder for join optimization
        # Actual implementation would depend on query structure
        return query
    
    def _add_buffer_hints(self, query: str) -> str:
        """Add buffer hints to query if supported."""
        # This is a placeholder for buffer hint implementation
        # Actual implementation would depend on database version
        return query

# Example usage
if __name__ == "__main__":
    try:
        # Initialize query optimizer
        optimizer = QueryOptimizer(
            stats_dir="query_stats",
            slow_query_threshold=1.0,
            cache_hit_threshold=0.95,
            max_parallel_queries=4
        )
        
        # Example query
        query = """
        SELECT s.timestamp, s.symbol, s.price, s.volume
        FROM snapshots s
        WHERE s.timestamp >= '2023-01-01'
        AND s.symbol = 'AAPL'
        ORDER BY s.timestamp DESC
        LIMIT 1000;
        """
        
        # Analyze query
        analysis = optimizer.analyze_query_plan(query)
        print("Query Analysis:", json.dumps(analysis, indent=2))
        
        # Get statistics
        stats = optimizer.get_query_stats()
        print("\nQuery Statistics:", json.dumps(stats, indent=2))
        
        # Optimize query
        optimized_query = optimizer.optimize_query(query)
        print("\nOptimized Query:", optimized_query)
        
    except Exception as e:
        print(f"Error: {str(e)}") 
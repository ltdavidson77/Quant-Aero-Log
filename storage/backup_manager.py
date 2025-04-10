# ==========================
# storage/backup_manager.py
# ==========================
# Comprehensive backup management with performance optimization.

import os
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from db_connector import execute_query, DatabaseError, QueryError
from utils.logger import get_logger
import json
import psutil

logger = get_logger("backup_manager")

class BackupManager:
    """Manages database backups with performance optimization."""
    
    def __init__(
        self,
        backup_dir: str = "backups",
        retention_days: int = 30,
        max_parallel_backups: int = 2,
        compression_level: int = 6
    ):
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.max_parallel_backups = max_parallel_backups
        self.compression_level = compression_level
        self.backup_metadata_file = self.backup_dir / "backup_metadata.json"
        self._setup_backup_directory()
        self._load_metadata()
    
    def _setup_backup_directory(self) -> None:
        """Setup backup directory structure."""
        try:
            # Create main backup directory
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different backup types
            (self.backup_dir / "full").mkdir(exist_ok=True)
            (self.backup_dir / "incremental").mkdir(exist_ok=True)
            (self.backup_dir / "wal").mkdir(exist_ok=True)
            
            logger.info(f"Backup directory structure created at {self.backup_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup backup directory: {str(e)}")
            raise
    
    def _load_metadata(self) -> None:
        """Load backup metadata from file."""
        try:
            if self.backup_metadata_file.exists():
                with open(self.backup_metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    'last_full_backup': None,
                    'last_incremental_backup': None,
                    'backup_chain': [],
                    'failed_backups': []
                }
                self._save_metadata()
                
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {str(e)}")
            raise
    
    def _save_metadata(self) -> None:
        """Save backup metadata to file."""
        try:
            with open(self.backup_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {str(e)}")
            raise
    
    def _check_system_resources(self) -> bool:
        """
        Check if system resources are sufficient for backup.
        
        Returns:
            bool: True if resources are sufficient, False otherwise
        """
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                logger.warning(f"High CPU usage ({cpu_percent}%), may affect backup performance")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                logger.warning(f"High memory usage ({memory.percent}%), may affect backup performance")
                return False
            
            # Check disk space
            backup_disk = psutil.disk_usage(str(self.backup_dir))
            if backup_disk.percent > 90:
                logger.warning(f"Low disk space ({backup_disk.percent}%), backup may fail")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check system resources: {str(e)}")
            return False
    
    def _perform_full_backup(self) -> str:
        """
        Perform a full database backup.
        
        Returns:
            str: Path to the backup file
        """
        try:
            if not self._check_system_resources():
                raise BackupError("Insufficient system resources for backup")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / "full" / f"full_backup_{timestamp}.sql.gz"
            
            # Use pg_dump with compression and performance optimizations
            cmd = [
                "pg_dump",
                "-h", os.getenv("PG_HOST", "localhost"),
                "-p", os.getenv("PG_PORT", "5432"),
                "-U", os.getenv("PG_USER", "postgres"),
                "-d", os.getenv("PG_DB", "quantdata"),
                "-F", "c",  # Custom format
                "-v",  # Verbose
                "-Z", str(self.compression_level),  # Compression level
                "-j", "4",  # Number of parallel jobs
                "-f", str(backup_file)
            ]
            
            # Set environment for password
            env = os.environ.copy()
            env["PGPASSWORD"] = os.getenv("PG_PASSWORD", "")
            
            # Execute backup
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Monitor backup progress
            while True:
                output = process.stderr.readline().decode()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"Backup progress: {output.strip()}")
            
            if process.returncode != 0:
                raise BackupError(f"Backup failed with return code {process.returncode}")
            
            # Update metadata
            self.metadata['last_full_backup'] = timestamp
            self.metadata['backup_chain'].append({
                'type': 'full',
                'timestamp': timestamp,
                'file': str(backup_file)
            })
            self._save_metadata()
            
            logger.info(f"Full backup completed: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Full backup failed: {str(e)}")
            self.metadata['failed_backups'].append({
                'type': 'full',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            self._save_metadata()
            raise
    
    def _perform_incremental_backup(self) -> str:
        """
        Perform an incremental backup.
        
        Returns:
            str: Path to the backup file
        """
        try:
            if not self._check_system_resources():
                raise BackupError("Insufficient system resources for backup")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / "incremental" / f"incremental_backup_{timestamp}.sql.gz"
            
            # Get the last backup timestamp
            last_backup = self.metadata['last_full_backup'] or self.metadata['last_incremental_backup']
            if not last_backup:
                raise BackupError("No previous backup found for incremental backup")
            
            # Use pg_dump with incremental options
            cmd = [
                "pg_dump",
                "-h", os.getenv("PG_HOST", "localhost"),
                "-p", os.getenv("PG_PORT", "5432"),
                "-U", os.getenv("PG_USER", "postgres"),
                "-d", os.getenv("PG_DB", "quantdata"),
                "-F", "c",
                "-v",
                "-Z", str(self.compression_level),
                "-j", "4",
                "--since", f"'{last_backup}'",
                "-f", str(backup_file)
            ]
            
            env = os.environ.copy()
            env["PGPASSWORD"] = os.getenv("PG_PASSWORD", "")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            while True:
                output = process.stderr.readline().decode()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"Backup progress: {output.strip()}")
            
            if process.returncode != 0:
                raise BackupError(f"Incremental backup failed with return code {process.returncode}")
            
            self.metadata['last_incremental_backup'] = timestamp
            self.metadata['backup_chain'].append({
                'type': 'incremental',
                'timestamp': timestamp,
                'file': str(backup_file)
            })
            self._save_metadata()
            
            logger.info(f"Incremental backup completed: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Incremental backup failed: {str(e)}")
            self.metadata['failed_backups'].append({
                'type': 'incremental',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            self._save_metadata()
            raise
    
    def _cleanup_old_backups(self) -> None:
        """Remove backups older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean full backups
            for backup_file in (self.backup_dir / "full").glob("*.sql.gz"):
                file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_date < cutoff_date:
                    backup_file.unlink()
                    logger.info(f"Removed old full backup: {backup_file}")
            
            # Clean incremental backups
            for backup_file in (self.backup_dir / "incremental").glob("*.sql.gz"):
                file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_date < cutoff_date:
                    backup_file.unlink()
                    logger.info(f"Removed old incremental backup: {backup_file}")
            
            # Clean WAL files
            for wal_file in (self.backup_dir / "wal").glob("*"):
                file_date = datetime.fromtimestamp(wal_file.stat().st_mtime)
                if file_date < cutoff_date:
                    wal_file.unlink()
                    logger.info(f"Removed old WAL file: {wal_file}")
            
            # Update metadata
            self.metadata['backup_chain'] = [
                backup for backup in self.metadata['backup_chain']
                if datetime.strptime(backup['timestamp'], "%Y%m%d_%H%M%S") >= cutoff_date
            ]
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {str(e)}")
            raise
    
    def schedule_backups(
        self,
        full_backup_interval: int = 7,  # days
        incremental_backup_interval: int = 1,  # days
        cleanup_interval: int = 1  # days
    ) -> None:
        """
        Schedule regular backups with performance considerations.
        
        Args:
            full_backup_interval: Days between full backups
            incremental_backup_interval: Days between incremental backups
            cleanup_interval: Days between cleanup operations
        """
        try:
            with ThreadPoolExecutor(max_workers=self.max_parallel_backups) as executor:
                # Schedule full backups
                if not self.metadata['last_full_backup'] or \
                   (datetime.now() - datetime.strptime(self.metadata['last_full_backup'], "%Y%m%d_%H%M%S")).days >= full_backup_interval:
                    executor.submit(self._perform_full_backup)
                
                # Schedule incremental backups
                elif not self.metadata['last_incremental_backup'] or \
                     (datetime.now() - datetime.strptime(self.metadata['last_incremental_backup'], "%Y%m%d_%H%M%S")).days >= incremental_backup_interval:
                    executor.submit(self._perform_incremental_backup)
                
                # Schedule cleanup
                executor.submit(self._cleanup_old_backups)
                
        except Exception as e:
            logger.error(f"Failed to schedule backups: {str(e)}")
            raise
    
    def restore_backup(self, backup_file: str, target_time: Optional[datetime] = None) -> None:
        """
        Restore database from backup with point-in-time recovery.
        
        Args:
            backup_file: Path to the backup file
            target_time: Optional point-in-time to restore to
        """
        try:
            if not self._check_system_resources():
                raise BackupError("Insufficient system resources for restore")
            
            # Stop any running database connections
            execute_query("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database();")
            
            # Restore from backup
            cmd = [
                "pg_restore",
                "-h", os.getenv("PG_HOST", "localhost"),
                "-p", os.getenv("PG_PORT", "5432"),
                "-U", os.getenv("PG_USER", "postgres"),
                "-d", os.getenv("PG_DB", "quantdata"),
                "-v",
                "-j", "4",  # Parallel restore
                "-c",  # Clean (drop) database objects before recreating
                "-1",  # Single transaction
                backup_file
            ]
            
            if target_time:
                cmd.extend(["--recovery-target-time", target_time.isoformat()])
            
            env = os.environ.copy()
            env["PGPASSWORD"] = os.getenv("PG_PASSWORD", "")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            while True:
                output = process.stderr.readline().decode()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    logger.info(f"Restore progress: {output.strip()}")
            
            if process.returncode != 0:
                raise BackupError(f"Restore failed with return code {process.returncode}")
            
            logger.info(f"Database restored from backup: {backup_file}")
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise

class BackupError(Exception):
    """Custom exception for backup-related errors."""
    pass

# Example usage
if __name__ == "__main__":
    try:
        # Initialize backup manager
        manager = BackupManager(
            backup_dir="backups",
            retention_days=30,
            max_parallel_backups=2,
            compression_level=6
        )
        
        # Schedule regular backups
        manager.schedule_backups(
            full_backup_interval=7,
            incremental_backup_interval=1,
            cleanup_interval=1
        )
        
        # Example restore
        # manager.restore_backup("backups/full/full_backup_20230101_000000.sql.gz")
        
    except Exception as e:
        print(f"Error: {str(e)}")

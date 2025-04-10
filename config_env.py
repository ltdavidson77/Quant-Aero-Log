# ==========================
# config_env.py
# ==========================
# Configuration and environment setup.

import os
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import dotenv
from datetime import datetime
from enum import Enum

# Add physics directory to path
sys.path.append(str(Path(__file__).parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

# Load .env file if present
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    DEV = "DEV"
    TEST = "TEST"
    PROD = "PROD"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

# -----------------------------
# Runtime Environment Flags
# -----------------------------
ENV: Environment = Environment(os.getenv("ENV", "DEV"))
USE_LIVE_FEED: bool = os.getenv("USE_LIVE_FEED", "False").lower() == "true"
ENABLE_DB_WRITE: bool = os.getenv("ENABLE_DB_WRITE", "True").lower() == "true"
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

# -----------------------------
# API Keys (optional, override)
# -----------------------------
API_KEYS: Dict[str, str] = {
    "ALPACA": {
        "key": os.getenv("ALPACA_API_KEY", ""),
        "secret": os.getenv("ALPACA_SECRET_KEY", "")
    },
    "BINANCE": {
        "key": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_SECRET_KEY", "")
    },
    "IBKR": {
        "client_id": os.getenv("IBKR_CLIENT_ID", "")
    },
    "COINBASE": {
        "key": os.getenv("COINBASE_API_KEY", ""),
        "secret": os.getenv("COINBASE_SECRET_KEY", "")
    }
}

# -----------------------------
# PostgreSQL Configuration
# -----------------------------
DB_CONFIG: Dict[str, Any] = {
    "database": os.getenv("PG_DB", "quantdb"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "password"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5432")),
    "pool_size": int(os.getenv("PG_POOL_SIZE", "5")),
    "max_overflow": int(os.getenv("PG_MAX_OVERFLOW", "10"))
}

# -----------------------------
# Paths and Directories
# -----------------------------
BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
PATHS: Dict[str, str] = {
    "data": os.path.join(BASE_DIR, "data"),
    "exports": os.path.join(BASE_DIR, "exports"),
    "models": os.path.join(BASE_DIR, "models"),
    "logs": os.path.join(BASE_DIR, "logs"),
    "cache": os.path.join(BASE_DIR, "cache")
}

# -----------------------------
# Thresholds and Constants
# -----------------------------
THRESHOLDS: Dict[str, float] = {
    "prediction": float(os.getenv("PREDICTION_THRESH", "0.3")),
    "correlation": float(os.getenv("CORRELATION_THRESH", "0.7")),
    "volatility": float(os.getenv("VOLATILITY_THRESH", "0.02"))
}

LABEL_HORIZONS: List[int] = [5, 15, 60, 1440]  # in minutes
SIGNAL_RESOLUTIONS: List[int] = [1, 5, 10, 60, 180, 300, 600, 1440, 2880, 10080, 43200]

# -----------------------------
# Snapshot Timing
# -----------------------------
SNAPSHOT_CONFIG: Dict[str, int] = {
    "interval_seconds": int(os.getenv("SNAPSHOT_INTERVAL_SECONDS", "300")),
    "retention_days": int(os.getenv("SNAPSHOT_RETENTION_DAYS", "30")),
    "max_snapshots": int(os.getenv("MAX_SNAPSHOTS", "1000"))
}

# -----------------------------
# Log Settings
# -----------------------------
LOG_CONFIG: Dict[str, Any] = {
    "level": LogLevel(os.getenv("LOG_LEVEL", "INFO")),
    "to_file": os.getenv("LOG_TO_FILE", "False").lower() == "true",
    "max_size_mb": int(os.getenv("LOG_MAX_SIZE_MB", "10")),
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5"))
}

# -----------------------------
# Debug Print
# -----------------------------
def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    if DEBUG_MODE:
        print("\n[CONFIG] Environment Summary:")
        print(f"  Environment: {ENV.value}")
        print(f"  Debug Mode: {DEBUG_MODE}")
        print(f"  Live Feed: {USE_LIVE_FEED}")
        print(f"  DB Write Enabled: {ENABLE_DB_WRITE}")
        
        print("\n[CONFIG] API Status:")
        for exchange, keys in API_KEYS.items():
            has_keys = any(keys.values())
            print(f"  {exchange}: {'Configured' if has_keys else 'Not Configured'}")
        
        print("\n[CONFIG] Database:")
        print(f"  Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        print(f"  Database: {DB_CONFIG['database']}")
        
        print("\n[CONFIG] Thresholds:")
        for name, value in THRESHOLDS.items():
            print(f"  {name.title()}: {value}")
        
        print("\n[CONFIG] Timing:")
        print(f"  Snapshot Interval: {SNAPSHOT_CONFIG['interval_seconds']}s")
        print(f"  Retention: {SNAPSHOT_CONFIG['retention_days']} days")
        
        print("\n[CONFIG] Logging:")
        print(f"  Level: {LOG_CONFIG['level'].value}")
        print(f"  To File: {LOG_CONFIG['to_file']}")

if DEBUG_MODE:
    print_config_summary()

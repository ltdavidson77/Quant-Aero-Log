# ==========================
# config.py
# ==========================
# Configuration loading utilities.

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        else:
            config_path = Path(config_path)
            
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {} 
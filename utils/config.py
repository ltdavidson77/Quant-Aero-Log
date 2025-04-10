# ==========================
# config.py
# ==========================
# Configuration loading utilities.

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import os
from dotenv import load_dotenv
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigManager:
    """Manages configuration loading and environment variables."""
    
    def __init__(self):
        self._config = {}
        self._env_vars = {}
        self._load_env_vars()
        
    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()
        self._env_vars = dict(os.environ)
        
    def _interpolate_env_vars(self, value: str) -> str:
        """Replace environment variable references in strings."""
        def replace_var(match):
            var_name = match.group(1)
            return self._env_vars.get(var_name, match.group(0))
            
        return re.sub(r'\${([^}]+)}', replace_var, value)
        
    def _process_value(self, value: Any) -> Any:
        """Process configuration values, handling environment variables."""
        if isinstance(value, str):
            return self._interpolate_env_vars(value)
        elif isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._process_value(v) for v in value]
        return value
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load and process configuration from file."""
        try:
            if config_path is None:
                config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
            else:
                config_path = Path(config_path)
                
            if not config_path.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")
                
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {config_path.suffix}")
                
            # Process configuration values
            processed_config = self._process_value(config)
            self._config.update(processed_config)
            return processed_config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise ConfigError(f"Failed to load configuration: {str(e)}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(config)
        
    def reload(self) -> None:
        """Reload configuration and environment variables."""
        self._load_env_vars()
        self.load_config()
        
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

# Create global config manager instance
config_manager = ConfigManager()

@lru_cache(maxsize=128)
def get_config(key: Optional[str] = None, default: Any = None) -> Union[Dict[str, Any], Any]:
    """Get configuration value with caching."""
    if key is None:
        return config_manager.config
    return config_manager.get(key, default)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    return config_manager.load_config(config_path)

def set_config(key: str, value: Any) -> None:
    """Set a configuration value."""
    config_manager.set(key, value)
    
def update_config(config: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    config_manager.update(config)
    
def reload_config() -> None:
    """Reload configuration and environment variables."""
    config_manager.reload() 
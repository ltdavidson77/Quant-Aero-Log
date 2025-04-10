# ==========================
# model_manager.py
# ==========================
# Manages model loading, saving, and versioning with quantum state integration.

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import logging
import json
import pickle
import hashlib
import os
import time
import gc
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime
import sys
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
import psutil
import signal
import asyncio
from collections import deque

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass

class ModelSaveError(ModelError):
    """Exception raised when model saving fails."""
    pass

class ModelVersionError(ModelError):
    """Exception raised when version-related operations fail."""
    pass

class ModelType(Enum):
    XGBOOST = auto()
    PYTORCH = auto()
    SKLEARN = auto()
    QUANTUM = auto()
    ENSEMBLE = auto()

@dataclass
class ModelMetadata:
    """Container for model metadata."""
    name: str
    version: str
    type: ModelType
    created_at: datetime
    last_updated: datetime
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    quantum_state: Optional[QuantumState] = None
    quantum_metadata: Optional[QuantumStateMetadata] = None
    performance_metrics: Dict[str, Any] = None
    dependencies: List[str] = None
    description: str = ""
    tags: List[str] = None

class ModelManager:
    def __init__(self, base_path: str = "models", max_models: int = 100):
        self.base_path = Path(base_path)
        self.max_models = max_models
        self.models = {}
        self.metadata = {}
        self.quantum_manager = QuantumStateManager()
        self.performance_monitor = PerformanceMonitor()
        self._ensure_directories()
        self._setup_signal_handlers()
        
    def _ensure_directories(self):
        """Ensure necessary directories exist."""
        try:
            self.base_path.mkdir(exist_ok=True)
            (self.base_path / "versions").mkdir(exist_ok=True)
            (self.base_path / "metadata").mkdir(exist_ok=True)
            (self.base_path / "cache").mkdir(exist_ok=True)
            (self.base_path / "logs").mkdir(exist_ok=True)
        except Exception as e:
            raise ModelError(f"Failed to create directories: {str(e)}")
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_terminate)
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {str(e)}")
        
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal."""
        logger.info("Received interrupt signal, saving state...")
        try:
            self._save_state()
        except Exception as e:
            logger.error(f"Error saving state during interrupt: {str(e)}")
        sys.exit(0)
        
    def _handle_terminate(self, signum, frame):
        """Handle terminate signal."""
        logger.info("Received terminate signal, saving state...")
        try:
            self._save_state()
        except Exception as e:
            logger.error(f"Error saving state during terminate: {str(e)}")
        sys.exit(0)
        
    def _save_state(self):
        """Save current state to disk."""
        try:
            state = {
                'models': {k: self._serialize_model(v) for k, v in self.models.items()},
                'metadata': {k: self._serialize_metadata(v) for k, v in self.metadata.items()}
            }
            
            state_path = self.base_path / "state.json"
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            raise ModelSaveError(f"Failed to save state: {str(e)}")
            
    def _load_state(self):
        """Load state from disk."""
        try:
            state_path = self.base_path / "state.json"
            if not state_path.exists():
                return
                
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            self.models = {k: self._deserialize_model(v) for k, v in state['models'].items()}
            self.metadata = {k: self._deserialize_metadata(v) for k, v in state['metadata'].items()}
        except Exception as e:
            raise ModelLoadError(f"Failed to load state: {str(e)}")
            
    def _serialize_model(self, model: Any) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        try:
            if isinstance(model, xgb.XGBClassifier):
                return {
                    'type': 'xgboost',
                    'params': model.get_params(),
                    'state': model.save_raw().decode()
                }
            elif isinstance(model, torch.nn.Module):
                return {
                    'type': 'pytorch',
                    'state_dict': {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
                }
            else:
                return {
                    'type': 'other',
                    'state': pickle.dumps(model).hex()
                }
        except Exception as e:
            raise ModelSaveError(f"Failed to serialize model: {str(e)}")
            
    def _deserialize_model(self, data: Dict[str, Any]) -> Any:
        """Deserialize model from dictionary."""
        try:
            if data['type'] == 'xgboost':
                model = xgb.XGBClassifier()
                model.load_model(data['state'])
                return model
            elif data['type'] == 'pytorch':
                model = torch.nn.Module()
                model.load_state_dict({k: torch.tensor(v) for k, v in data['state_dict'].items()})
                return model
            else:
                return pickle.loads(bytes.fromhex(data['state']))
        except Exception as e:
            raise ModelLoadError(f"Failed to deserialize model: {str(e)}")
            
    def _serialize_metadata(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        try:
            return {
                'name': metadata.name,
                'version': metadata.version,
                'type': metadata.type.name,
                'created_at': metadata.created_at.isoformat(),
                'last_updated': metadata.last_updated.isoformat(),
                'metrics': metadata.metrics,
                'params': metadata.params,
                'quantum_state': self.quantum_manager.serialize_state(metadata.quantum_state) if metadata.quantum_state else None,
                'quantum_metadata': self.quantum_manager.serialize_metadata(metadata.quantum_metadata) if metadata.quantum_metadata else None,
                'performance_metrics': metadata.performance_metrics,
                'dependencies': metadata.dependencies,
                'description': metadata.description,
                'tags': metadata.tags
            }
        except Exception as e:
            raise ModelSaveError(f"Failed to serialize metadata: {str(e)}")
        
    def _deserialize_metadata(self, data: Dict[str, Any]) -> ModelMetadata:
        """Deserialize metadata from dictionary."""
        try:
            return ModelMetadata(
                name=data['name'],
                version=data['version'],
                type=ModelType[data['type']],
                created_at=datetime.fromisoformat(data['created_at']),
                last_updated=datetime.fromisoformat(data['last_updated']),
                metrics=data['metrics'],
                params=data['params'],
                quantum_state=self.quantum_manager.deserialize_state(data['quantum_state']) if data['quantum_state'] else None,
                quantum_metadata=self.quantum_manager.deserialize_metadata(data['quantum_metadata']) if data['quantum_metadata'] else None,
                performance_metrics=data['performance_metrics'],
                dependencies=data['dependencies'],
                description=data['description'],
                tags=data['tags']
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to deserialize metadata: {str(e)}")
        
    def save_model(self, model: Any, name: str, version: str,
                  metrics: Optional[Dict[str, Any]] = None,
                  params: Optional[Dict[str, Any]] = None,
                  quantum_state: Optional[QuantumState] = None,
                  quantum_metadata: Optional[QuantumStateMetadata] = None,
                  description: str = "",
                  tags: Optional[List[str]] = None) -> str:
        """Save model with metadata and quantum state."""
        try:
            if len(self.models) >= self.max_models:
                self._evict_oldest_model()
                
            model_type = self._determine_model_type(model)
            now = datetime.now()
            
            metadata = ModelMetadata(
                name=name,
                version=version,
                type=model_type,
                created_at=now,
                last_updated=now,
                metrics=metrics or {},
                params=params or {},
                quantum_state=quantum_state,
                quantum_metadata=quantum_metadata,
                performance_metrics={},
                dependencies=self._get_model_dependencies(model),
                description=description,
                tags=tags or []
            )
            
            self.models[f"{name}_{version}"] = model
            self.metadata[f"{name}_{version}"] = metadata
            
            self._save_state()
            logger.info(f"Saved model {name} version {version}")
            return f"{name}_{version}"
        except Exception as e:
            raise ModelSaveError(f"Failed to save model {name} version {version}: {str(e)}")
        
    def load_model(self, name: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """Load model and its metadata."""
        try:
            if version is None:
                version = self._get_latest_version(name)
                
            key = f"{name}_{version}"
            if key not in self.models:
                raise ModelLoadError(f"Model {name} version {version} not found")
                
            model = self.models[key]
            metadata = self.metadata[key]
            
            # Update access metrics
            metadata.last_updated = datetime.now()
            self.performance_monitor.update_access(key)
            
            return model, metadata
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {name} version {version}: {str(e)}")
        
    def _determine_model_type(self, model: Any) -> ModelType:
        """Determine model type."""
        try:
            if isinstance(model, xgb.XGBClassifier):
                return ModelType.XGBOOST
            elif isinstance(model, torch.nn.Module):
                return ModelType.PYTORCH
            elif hasattr(model, 'fit') and hasattr(model, 'predict'):
                return ModelType.SKLEARN
            elif hasattr(model, 'quantum_state'):
                return ModelType.QUANTUM
            else:
                return ModelType.ENSEMBLE
        except Exception as e:
            raise ModelError(f"Failed to determine model type: {str(e)}")
            
    def _get_model_dependencies(self, model: Any) -> List[str]:
        """Get model dependencies."""
        try:
            dependencies = []
            
            if isinstance(model, xgb.XGBClassifier):
                dependencies.extend(['xgboost', 'numpy'])
            elif isinstance(model, torch.nn.Module):
                dependencies.extend(['torch', 'numpy'])
            elif hasattr(model, 'quantum_state'):
                dependencies.extend(['quantum', 'numpy'])
                
            return dependencies
        except Exception as e:
            raise ModelError(f"Failed to get model dependencies: {str(e)}")
            
    def _get_latest_version(self, name: str) -> str:
        """Get latest version of a model."""
        try:
            versions = [v.split('_')[1] for k, v in self.metadata.items() if k.startswith(name)]
            if not versions:
                raise ModelVersionError(f"No versions found for model {name}")
            return max(versions)
        except Exception as e:
            raise ModelVersionError(f"Failed to get latest version for model {name}: {str(e)}")
            
    def _evict_oldest_model(self):
        """Evict oldest model to make space."""
        try:
            oldest_key = min(self.metadata.items(), key=lambda x: x[1].last_updated)[0]
            del self.models[oldest_key]
            del self.metadata[oldest_key]
            logger.info(f"Evicted model {oldest_key}")
        except Exception as e:
            raise ModelError(f"Failed to evict oldest model: {str(e)}")
            
    def get_model_info(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model."""
        try:
            if version is None:
                version = self._get_latest_version(name)
                
            key = f"{name}_{version}"
            if key not in self.metadata:
                raise ModelError(f"Model {name} version {version} not found")
                
            metadata = self.metadata[key]
            return {
                'name': metadata.name,
                'version': metadata.version,
                'type': metadata.type.name,
                'created_at': metadata.created_at,
                'last_updated': metadata.last_updated,
                'metrics': metadata.metrics,
                'params': metadata.params,
                'performance_metrics': metadata.performance_metrics,
                'dependencies': metadata.dependencies,
                'description': metadata.description,
                'tags': metadata.tags
            }
        except Exception as e:
            raise ModelError(f"Failed to get model info for {name} version {version}: {str(e)}")
            
    def delete_model(self, name: str, version: str):
        """Delete a model version."""
        try:
            key = f"{name}_{version}"
            if key not in self.models:
                raise ModelError(f"Model {name} version {version} not found")
                
            del self.models[key]
            del self.metadata[key]
            self._save_state()
            logger.info(f"Deleted model {name} version {version}")
        except Exception as e:
            raise ModelError(f"Failed to delete model {name} version {version}: {str(e)}")
            
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models."""
        try:
            return [
                {
                    'name': m.name,
                    'version': m.version,
                    'type': m.type.name,
                    'created_at': m.created_at,
                    'last_updated': m.last_updated
                }
                for m in self.metadata.values()
            ]
        except Exception as e:
            raise ModelError(f"Failed to list models: {str(e)}")
            
    def update_model_metrics(self, name: str, version: str, metrics: Dict[str, Any]):
        """Update model metrics."""
        try:
            key = f"{name}_{version}"
            if key not in self.metadata:
                raise ModelError(f"Model {name} version {version} not found")
                
            self.metadata[key].metrics.update(metrics)
            self.metadata[key].last_updated = datetime.now()
            self._save_state()
        except Exception as e:
            raise ModelError(f"Failed to update metrics for model {name} version {version}: {str(e)}")
            
    def update_quantum_state(self, name: str, version: str,
                           quantum_state: QuantumState,
                           quantum_metadata: Optional[QuantumStateMetadata] = None):
        """Update quantum state for a model."""
        try:
            key = f"{name}_{version}"
            if key not in self.metadata:
                raise ModelError(f"Model {name} version {version} not found")
                
            self.metadata[key].quantum_state = quantum_state
            self.metadata[key].quantum_metadata = quantum_metadata
            self.metadata[key].last_updated = datetime.now()
            self._save_state()
        except Exception as e:
            raise ModelError(f"Failed to update quantum state for model {name} version {version}: {str(e)}")

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.access_times = {}
        self.execution_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.gpu_usage = deque(maxlen=window_size)
        self.error_rates = deque(maxlen=window_size)
        
    def update_access(self, model_key: str):
        """Update access time for a model."""
        try:
            self.access_times[model_key] = time.time()
        except Exception as e:
            logger.error(f"Failed to update access time: {str(e)}")
            
    def update_execution(self, execution_time: float):
        """Update execution time metrics."""
        try:
            self.execution_times.append(execution_time)
        except Exception as e:
            logger.error(f"Failed to update execution time: {str(e)}")
            
    def update_resource_usage(self):
        """Update resource usage metrics."""
        try:
            self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
            self.cpu_usage.append(psutil.cpu_percent())
            if torch.cuda.is_available():
                self.gpu_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
        except Exception as e:
            logger.error(f"Failed to update resource usage: {str(e)}")
            
    def update_error(self, error: bool):
        """Update error metrics."""
        try:
            self.error_rates.append(1.0 if error else 0.0)
        except Exception as e:
            logger.error(f"Failed to update error metrics: {str(e)}")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            return {
                'avg_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
                'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
                'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                'avg_gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else 0,
                'error_rate': np.mean(self.error_rates) if self.error_rates else 0
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return {}
            
    def get_trends(self) -> Dict[str, List[float]]:
        """Get performance trends."""
        try:
            return {
                'execution_times': list(self.execution_times),
                'memory_usage': list(self.memory_usage),
                'cpu_usage': list(self.cpu_usage),
                'gpu_usage': list(self.gpu_usage),
                'error_rates': list(self.error_rates)
            }
        except Exception as e:
            logger.error(f"Failed to get trends: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize model manager
    manager = ModelManager()
    
    # Create sample model
    model = xgb.XGBClassifier()
    
    # Save model with metadata
    version = manager.save_model(
        model,
        name="test_model",
        version="1.0.0",
        metrics={"accuracy": 0.95},
        params={"max_depth": 3},
        description="Test model for demonstration",
        tags=["test", "demo"]
    )
    
    # Load model
    loaded_model, metadata = manager.load_model("test_model", "1.0.0")
    
    # Get model info
    info = manager.get_model_info("test_model", "1.0.0")
    print(f"Model info: {info}")
    
    # List all models
    models = manager.list_models()
    print(f"All models: {models}") 
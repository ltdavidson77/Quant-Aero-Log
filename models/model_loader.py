# ==========================
# models/model_loader.py
# ==========================
# Advanced model persistence and versioning.

import xgboost as xgb
import torch
import joblib
import os
import json
import logging
from typing import Dict, Any, Union, Optional
from datetime import datetime
import hashlib
import shutil
from pathlib import Path
import sys

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

# Import model management
from .model_manager import ModelManager, ModelError, ModelSaveError, ModelLoadError

logger = logging.getLogger(__name__)

class ModelLoaderError(Exception):
    """Base exception for model loading errors."""
    pass

class ModelVersionError(ModelLoaderError):
    """Exception raised when version management fails."""
    pass

class ModelPersistenceError(ModelLoaderError):
    """Exception raised when model persistence operations fail."""
    pass

class ModelVersion:
    def __init__(self, version: str, timestamp: str, metrics: Dict[str, Any],
                 params: Dict[str, Any], model_hash: str):
        try:
            self.version = version
            self.timestamp = timestamp
            self.metrics = metrics
            self.params = params
            self.model_hash = model_hash
        except Exception as e:
            raise ModelVersionError(f"Failed to initialize ModelVersion: {str(e)}")

class ModelLoader:
    def __init__(self, model_manager: ModelManager):
        try:
            self.model_manager = model_manager
            self.version_file = os.path.join(model_manager.base_path, "versions.json")
            self._ensure_directories()
        except Exception as e:
            raise ModelLoaderError(f"Failed to initialize ModelLoader: {str(e)}")
        
    def _ensure_directories(self):
        """Ensure necessary directories exist."""
        try:
            os.makedirs(self.model_manager.base_path, exist_ok=True)
            os.makedirs(os.path.join(self.model_manager.base_path, "versions"), exist_ok=True)
            os.makedirs(os.path.join(self.model_manager.base_path, "metadata"), exist_ok=True)
        except Exception as e:
            raise ModelPersistenceError(f"Failed to ensure directories: {str(e)}")
        
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA-256 hash of model file."""
        try:
            with open(model_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            raise ModelPersistenceError(f"Failed to calculate model hash: {str(e)}")
            
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load version history from JSON file."""
        try:
            if not os.path.exists(self.version_file):
                return {}
                
            with open(self.version_file, 'r') as f:
                versions_data = json.load(f)
                
            return {
                version: ModelVersion(**data)
                for version, data in versions_data.items()
            }
        except Exception as e:
            raise ModelVersionError(f"Failed to load versions: {str(e)}")
        
    def _save_versions(self, versions: Dict[str, ModelVersion]):
        """Save version history to JSON file."""
        try:
            versions_data = {
                version: {
                    'version': v.version,
                    'timestamp': v.timestamp,
                    'metrics': v.metrics,
                    'params': v.params,
                    'model_hash': v.model_hash
                }
                for version, v in versions.items()
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(versions_data, f, indent=2)
        except Exception as e:
            raise ModelVersionError(f"Failed to save versions: {str(e)}")
            
    def save_model(self, model: Union[xgb.XGBClassifier, torch.nn.Module],
                  model_path: str,
                  metrics: Optional[Dict[str, Any]] = None,
                  params: Optional[Dict[str, Any]] = None) -> str:
        """Save model with versioning and metadata."""
        try:
            # Generate version string
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
            
            # Create version-specific paths
            version_path = os.path.join(self.model_manager.base_path, "versions", version)
            os.makedirs(version_path, exist_ok=True)
            
            # Save model
            if isinstance(model, xgb.XGBClassifier):
                model_path = os.path.join(version_path, "model.json")
                model.save_model(model_path)
            elif isinstance(model, torch.nn.Module):
                model_path = os.path.join(version_path, "model.pt")
                torch.save(model.state_dict(), model_path)
            else:
                model_path = os.path.join(version_path, "model.joblib")
                joblib.dump(model, model_path)
                
            # Calculate model hash
            model_hash = self._calculate_model_hash(model_path)
            
            # Save metadata
            metadata = {
                'version': version,
                'timestamp': timestamp,
                'metrics': metrics or {},
                'params': params or {},
                'model_hash': model_hash,
                'model_type': model.__class__.__name__
            }
            
            metadata_path = os.path.join(self.model_manager.base_path, "metadata", f"{version}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Update versions
            versions = self._load_versions()
            versions[version] = ModelVersion(
                version=version,
                timestamp=timestamp,
                metrics=metrics or {},
                params=params or {},
                model_hash=model_hash
            )
            self._save_versions(versions)
            
            # Create symlink to latest version
            latest_path = os.path.join(self.model_manager.base_path, "latest")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(version_path, latest_path)
            
            logger.info(f"Model saved as version {version}")
            return version
        except Exception as e:
            raise ModelSaveError(f"Failed to save model: {str(e)}")
        
    def load_model(self, version: Optional[str] = None) -> Union[xgb.XGBClassifier, torch.nn.Module]:
        """Load model by version or latest version."""
        try:
            if version is None:
                version_path = os.path.join(self.model_manager.base_path, "latest")
                if not os.path.exists(version_path):
                    raise FileNotFoundError("No model versions found")
                version = os.path.basename(os.path.realpath(version_path))
            else:
                version_path = os.path.join(self.model_manager.base_path, "versions", version)
                
            if not os.path.exists(version_path):
                raise FileNotFoundError(f"Version {version} not found")
                
            # Load metadata
            metadata_path = os.path.join(self.model_manager.base_path, "metadata", f"{version}.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Load model based on type
            model_type = metadata['model_type']
            model_path = os.path.join(version_path, "model.json" if model_type == "XGBClassifier" else "model.pt")
            
            if model_type == "XGBClassifier":
                model = xgb.XGBClassifier()
                model.load_model(model_path)
            elif model_type.startswith("Neural"):
                model = torch.load(model_path)
            else:
                model = joblib.load(model_path)
                
            logger.info(f"Loaded model version {version}")
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")
        
    def get_version_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific version or all versions."""
        try:
            versions = self._load_versions()
            
            if version is None:
                return {
                    'versions': list(versions.keys()),
                    'latest': os.path.basename(os.path.realpath(os.path.join(self.model_manager.base_path, "latest")))
                }
                
            if version not in versions:
                raise ValueError(f"Version {version} not found")
                
            return {
                'version': versions[version].version,
                'timestamp': versions[version].timestamp,
                'metrics': versions[version].metrics,
                'params': versions[version].params,
                'model_hash': versions[version].model_hash
            }
        except Exception as e:
            raise ModelVersionError(f"Failed to get version info: {str(e)}")
        
    def delete_version(self, version: str):
        """Delete a specific model version."""
        try:
            versions = self._load_versions()
            
            if version not in versions:
                raise ValueError(f"Version {version} not found")
                
            # Remove version directory
            version_path = os.path.join(self.model_manager.base_path, "versions", version)
            if os.path.exists(version_path):
                shutil.rmtree(version_path)
                
            # Remove metadata
            metadata_path = os.path.join(self.model_manager.base_path, "metadata", f"{version}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            # Update versions
            del versions[version]
            self._save_versions(versions)
            
            logger.info(f"Deleted model version {version}")
        except Exception as e:
            raise ModelVersionError(f"Failed to delete version: {str(e)}")

def save_model(model: Union[xgb.XGBClassifier, torch.nn.Module],
              model_path: str,
              metrics: Optional[Dict[str, Any]] = None,
              params: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function for saving models."""
    try:
        manager = ModelManager(os.path.dirname(model_path))
        loader = ModelLoader(manager)
        return loader.save_model(model, model_path, metrics, params)
    except Exception as e:
        raise ModelSaveError(f"Failed to save model: {str(e)}")

def load_model(model_path: str, version: Optional[str] = None) -> Union[xgb.XGBClassifier, torch.nn.Module]:
    """Convenience function for loading models."""
    try:
        manager = ModelManager(os.path.dirname(model_path))
        loader = ModelLoader(manager)
        return loader.load_model(version)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize model manager and loader
        manager = ModelManager()
        loader = ModelLoader(manager)
        
        # Create sample model
        model = xgb.XGBClassifier()
        
        # Save model with metadata
        version = loader.save_model(
            model,
            "models/xgb_model.json",
            metrics={"accuracy": 0.95},
            params={"max_depth": 3}
        )
        
        # Load model
        loaded_model = loader.load_model(version)
        
        # Get version info
        info = loader.get_version_info(version)
        print(f"Version info: {info}")
        
        # List all versions
        all_versions = loader.get_version_info()
        print(f"All versions: {all_versions}")
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise

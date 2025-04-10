# ==========================
# models/training_pipeline.py
# ==========================
# Full model training logic with persistence and advanced features.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_loader import save_model
from evaluation_metrics import evaluate_classification
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
import optuna
from concurrent.futures import ThreadPoolExecutor
import time
import joblib
from dataclasses import dataclass
from pathlib import Path
import sys
import gc
import signal
import asyncio
from collections import deque
import json
import pickle
import hashlib
from datetime import datetime

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

# Import model manager
from models.model_manager import ModelManager, ModelType, ModelMetadata, PerformanceMonitor, ModelError, ModelSaveError, ModelLoadError

logger = logging.getLogger(__name__)

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass

class DataPreparationError(TrainingError):
    """Exception raised when data preparation fails."""
    pass

class HyperparameterError(TrainingError):
    """Exception raised when hyperparameter optimization fails."""
    pass

class TrainingStateError(TrainingError):
    """Exception raised when training state management fails."""
    pass

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    n_trials: int = 100
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    eval_metric: str = 'logloss'
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    quantum_state_type: QuantumStateType = QuantumStateType.SUPERPOSITION
    quantum_state_params: Dict[str, Any] = None
    batch_size: int = 1024
    max_memory_usage: float = 0.8
    checkpoint_frequency: int = 10
    save_best_only: bool = True
    use_gpu: bool = True
    compression_level: int = 3
    encryption_key: Optional[str] = None

class AdvancedXGBoostTrainer:
    def __init__(self, model_manager: ModelManager, config: Optional[TrainingConfig] = None):
        try:
            self.model_manager = model_manager
            self.config = config or TrainingConfig()
            self.quantum_manager = QuantumStateManager()
            self.performance_monitor = PerformanceMonitor()
            self._setup_signal_handlers()
            self._load_training_state()
        except Exception as e:
            raise TrainingError(f"Failed to initialize trainer: {str(e)}")
        
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
            self._save_training_state()
        except Exception as e:
            logger.error(f"Error saving state during interrupt: {str(e)}")
        sys.exit(0)
        
    def _handle_terminate(self, signum, frame):
        """Handle terminate signal."""
        logger.info("Received terminate signal, saving state...")
        try:
            self._save_training_state()
        except Exception as e:
            logger.error(f"Error saving state during terminate: {str(e)}")
        sys.exit(0)
        
    def _save_training_state(self):
        """Save current training state."""
        try:
            state = {
                'config': self.config.__dict__,
                'best_params': self.best_params if hasattr(self, 'best_params') else None,
                'best_score': self.best_score if hasattr(self, 'best_score') else None,
                'trial_history': self.trial_history if hasattr(self, 'trial_history') else None
            }
            
            state_path = Path(self.model_manager.base_path) / "training_state.json"
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            raise TrainingStateError(f"Failed to save training state: {str(e)}")
            
    def _load_training_state(self):
        """Load training state from disk."""
        try:
            state_path = Path(self.model_manager.base_path) / "training_state.json"
            if not state_path.exists():
                return
                
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            self.config = TrainingConfig(**state['config'])
            self.best_params = state['best_params']
            self.best_score = state['best_score']
            self.trial_history = state['trial_history']
        except Exception as e:
            raise TrainingStateError(f"Failed to load training state: {str(e)}")
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training with quantum state integration."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.config.validation_size, random_state=self.config.random_state
            )
            
            # Initialize quantum state
            quantum_state = self.quantum_manager.create_state(
                self.config.quantum_state_type,
                self.config.quantum_state_params or {}
            )
            
            # Apply quantum state to data
            X_train = self._apply_quantum_state(X_train, quantum_state)
            X_val = self._apply_quantum_state(X_val, quantum_state)
            X_test = self._apply_quantum_state(X_test, quantum_state)
            
            return X_train, y_train, X_val, y_val, X_test, y_test
        except Exception as e:
            raise DataPreparationError(f"Failed to prepare data: {str(e)}")
        
    def _apply_quantum_state(self, X: pd.DataFrame, quantum_state: QuantumState) -> pd.DataFrame:
        """Apply quantum state to data."""
        try:
            # Convert to numpy for quantum operations
            X_np = X.values
            
            # Apply quantum state
            X_transformed = self.quantum_manager.apply_state(X_np, quantum_state)
            
            # Convert back to DataFrame
            return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
        except Exception as e:
            raise DataPreparationError(f"Failed to apply quantum state: {str(e)}")
            
    def _objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Objective function for hyperparameter optimization."""
        try:
            # Define hyperparameter space
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5)
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
            
            # Evaluate model
            y_pred = model.predict(X_val)
            score = evaluate_classification(y_val, y_pred)
            
            return score['f1_score']
        except Exception as e:
            raise HyperparameterError(f"Failed to evaluate hyperparameters: {str(e)}")
            
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        try:
            # Prepare data
            X_train, y_train, X_val, y_val, _, _ = self._prepare_data(X, y)
            
            # Create study
            study = optuna.create_study(direction='maximize')
            
            # Optimize
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.config.n_trials,
                n_jobs=self.config.n_jobs
            )
            
            # Store results
            self.best_params = study.best_params
            self.best_score = study.best_value
            self.trial_history = study.trials_dataframe().to_dict()
            
            return self.best_params
        except Exception as e:
            raise HyperparameterError(f"Failed to optimize hyperparameters: {str(e)}")
            
    def train_model(self, X: pd.DataFrame, y: pd.Series, name: str, version: str,
                   metrics: Optional[Dict[str, Any]] = None,
                   description: str = "",
                   tags: Optional[List[str]] = None) -> xgb.XGBClassifier:
        """Train model with optimized hyperparameters."""
        try:
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_data(X, y)
            
            # Optimize hyperparameters if not already done
            if not hasattr(self, 'best_params'):
                self.optimize_hyperparameters(X, y)
            
            # Create and train model
            model = xgb.XGBClassifier(**self.best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
            
            # Evaluate model
            y_pred = model.predict(X_test)
            model_metrics = evaluate_classification(y_test, y_pred)
            
            # Update metrics
            if metrics:
                model_metrics.update(metrics)
            
            # Save model
            self.model_manager.save_model(
                model,
                name=name,
                version=version,
                metrics=model_metrics,
                params=self.best_params,
                description=description,
                tags=tags
            )
            
            return model
        except Exception as e:
            raise TrainingError(f"Failed to train model: {str(e)}")
            
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics and history."""
        try:
            return {
                'best_params': self.best_params if hasattr(self, 'best_params') else None,
                'best_score': self.best_score if hasattr(self, 'best_score') else None,
                'trial_history': self.trial_history if hasattr(self, 'trial_history') else None,
                'performance_metrics': self.performance_monitor.get_metrics()
            }
        except Exception as e:
            raise TrainingError(f"Failed to get training metrics: {str(e)}")

def train_model(feature_df: pd.DataFrame, labels: pd.Series, 
                model_path: str = "models/xgb_model.json") -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Convenience function for training a model."""
    try:
        # Initialize model manager and trainer
        model_manager = ModelManager()
        trainer = AdvancedXGBoostTrainer(model_manager)
        
        # Train model
        model = trainer.train_model(
            feature_df,
            labels,
            name="xgb_model",
            version="1.0.0"
        )
        
        # Get metrics
        metrics = trainer.get_training_metrics()
        
        return model, metrics
    except Exception as e:
        raise TrainingError(f"Failed to train model: {str(e)}")


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
from typing import Tuple, Dict, Any
import optuna
from concurrent.futures import ThreadPoolExecutor
import time
import joblib

class AdvancedXGBoostTrainer:
    def __init__(self, use_gpu: bool = True, n_trials: int = 50):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.n_trials = n_trials
        self.best_params = None
        self.scaler = StandardScaler()
        
    def _prepare_data(self, feature_df: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and preprocess data for training."""
        feature_df = feature_df.fillna(method="bfill").fillna(method="ffill")
        labels = labels.loc[feature_df.index]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(feature_df)
        return X_scaled, labels.values
        
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function for hyperparameter optimization."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
        }
        
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            **params
        )
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = evaluate_classification(y_val, y_pred)[1]['Accuracy']
            scores.append(score)
            
        return np.mean(scores)
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)
        self.best_params = study.best_params
        return self.best_params
        
    def train_model(self, feature_df: pd.DataFrame, labels: pd.Series, 
                   model_path: str = "models/xgb_model.json") -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Train XGBoost model with advanced features."""
        start_time = time.time()
        
        # Prepare data
        X, y = self._prepare_data(feature_df, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Optimize hyperparameters
        if not self.best_params:
            logging.info("Optimizing hyperparameters...")
            self.optimize_hyperparameters(X_train, y_train)
            
        # Train final model
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            **self.best_params
        )
        
        if self.use_gpu:
            model.set_params(tree_method='gpu_hist', gpu_id=0)
            
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=True
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        metrics_df, summary = evaluate_classification(y_test, y_pred)
        
        # Save model and scaler
        save_model(model, model_path)
        joblib.dump(self.scaler, f"{model_path}.scaler")
        
        execution_time = time.time() - start_time
        logging.info(f"Training completed in {execution_time:.2f} seconds")
        logging.info(f"Model saved to {model_path}")
        
        return model, {
            "metrics": metrics_df,
            "summary": summary,
            "execution_time": execution_time,
            "best_params": self.best_params
        }

def train_model(feature_df: pd.DataFrame, labels: pd.Series, 
                model_path: str = "models/xgb_model.json") -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Main training function with default configuration."""
    trainer = AdvancedXGBoostTrainer()
    return trainer.train_model(feature_df, labels, model_path)


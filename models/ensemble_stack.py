# ==========================
# ensemble_stack.py
# ==========================
# Advanced ensemble methods for model combination.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import logging
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor
import time
import gc
from pathlib import Path
import sys

# Add physics directory to path
sys.path.append(str(Path(__file__).parent.parent / 'physics'))

# Import quantum state management
from physics.quantum_state import QuantumState
from physics.quantum_state_manager import QuantumStateManager, QuantumStateType, QuantumStateMetadata

logger = logging.getLogger(__name__)

class NeuralStacker(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(NeuralStacker, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

class AdvancedEnsemble:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.meta_model = None
        self.scaler = None
        
    def _prepare_meta_features(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Prepare meta-features from base model predictions."""
        stacked = np.stack(predictions_list, axis=1)
        return stacked
        
    def _train_neural_stacker(self, X: np.ndarray, y: np.ndarray) -> NeuralStacker:
        """Train neural network meta-learner."""
        model = NeuralStacker(input_size=X.shape[1])
        if self.use_gpu:
            model = model.cuda()
            
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        if self.use_gpu:
            X_tensor = X_tensor.cuda()
            y_tensor = y_tensor.cuda()
            
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(100):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
        return model
        
    def majority_vote(self, predictions_list: List[np.ndarray]) -> pd.Series:
        """Majority voting ensemble."""
    stacked = np.stack(predictions_list, axis=1)
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=stacked
        )
    return pd.Series(final_preds)

    def weighted_soft_voting(self, probabilities_list: List[np.ndarray], 
                           weights: np.ndarray = None) -> pd.Series:
        """Weighted soft voting ensemble."""
        stacked = np.stack(probabilities_list, axis=0)
    if weights is None:
        weights = np.ones(stacked.shape[0])
    weighted_avg = np.tensordot(weights, stacked, axes=(0, 0)) / np.sum(weights)
    final_preds = np.argmax(weighted_avg, axis=1)
    return pd.Series(final_preds)

    def stacking(self, predictions_list: List[np.ndarray], 
                y_true: np.ndarray, 
                method: str = 'logistic') -> pd.Series:
        """Stacking ensemble with multiple meta-learner options."""
        meta_features = self._prepare_meta_features(predictions_list)
        
        if method == 'logistic':
            meta_model = LogisticRegression()
        elif method == 'random_forest':
            meta_model = RandomForestClassifier()
        elif method == 'xgboost':
            meta_model = xgb.XGBClassifier()
        elif method == 'neural':
            meta_model = self._train_neural_stacker(meta_features, y_true)
        else:
            raise ValueError(f"Unknown stacking method: {method}")
            
        if method != 'neural':
            meta_model.fit(meta_features, y_true)
            final_preds = meta_model.predict(meta_features)
        else:
            meta_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(meta_features)
                if self.use_gpu:
                    X_tensor = X_tensor.cuda()
                outputs = meta_model(X_tensor)
                final_preds = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
                
        return pd.Series(final_preds)
        
    def dynamic_weighting(self, predictions_list: List[np.ndarray],
                         y_true: np.ndarray,
                         window_size: int = 20) -> pd.Series:
        """Dynamic weighting based on recent performance."""
        n_models = len(predictions_list)
        weights = np.ones(n_models) / n_models
        final_preds = []
        
        for i in range(len(y_true)):
            if i >= window_size:
                # Calculate recent performance
                recent_true = y_true[i-window_size:i]
                recent_preds = [pred[i-window_size:i] for pred in predictions_list]
                accuracies = [np.mean(pred == recent_true) for pred in recent_preds]
                
                # Update weights
                weights = np.array(accuracies)
                weights = weights / np.sum(weights)
                
            # Weighted prediction
            current_preds = np.array([pred[i] for pred in predictions_list])
            weighted_pred = np.average(current_preds, weights=weights)
            final_preds.append(weighted_pred)
            
        return pd.Series(final_preds)
        
    def ensemble_predict(self, predictions_list: List[np.ndarray],
                        y_true: np.ndarray = None,
                        method: str = 'stacking',
                        **kwargs) -> pd.Series:
        """Main ensemble prediction method."""
        start_time = time.time()
        
        if method == 'majority_vote':
            result = self.majority_vote(predictions_list)
        elif method == 'weighted_soft_voting':
            result = self.weighted_soft_voting(predictions_list, **kwargs)
        elif method == 'stacking':
            result = self.stacking(predictions_list, y_true, **kwargs)
        elif method == 'dynamic_weighting':
            result = self.dynamic_weighting(predictions_list, y_true, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
            
        execution_time = time.time() - start_time
        logging.info(f"Ensemble prediction completed in {execution_time:.2f} seconds")
        
        return result

def majority_vote(predictions_list: List[np.ndarray]) -> pd.Series:
    """Convenience function for majority voting."""
    ensemble = AdvancedEnsemble()
    return ensemble.majority_vote(predictions_list)

def weighted_soft_voting(probabilities_list: List[np.ndarray], 
                        weights: np.ndarray = None) -> pd.Series:
    """Convenience function for weighted soft voting."""
    ensemble = AdvancedEnsemble()
    return ensemble.weighted_soft_voting(probabilities_list, weights)

# Example usage
if __name__ == "__main__":
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 100
    n_models = 3
    
    predictions_list = [
        np.random.randint(0, 3, size=n_samples) for _ in range(n_models)
    ]
    y_true = np.random.randint(0, 3, size=n_samples)
    
    # Test different ensemble methods
    ensemble = AdvancedEnsemble()
    
    print("Majority Vote:")
    print(ensemble.majority_vote(predictions_list).head())
    
    print("\nWeighted Soft Voting:")
    print(ensemble.weighted_soft_voting(predictions_list).head())
    
    print("\nStacking (Logistic):")
    print(ensemble.stacking(predictions_list, y_true, method='logistic').head())
    
    print("\nDynamic Weighting:")
    print(ensemble.dynamic_weighting(predictions_list, y_true).head())

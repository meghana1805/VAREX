"""
Model definitions and training logic for MAGPIE-Lite.
"""
import os
import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict, model_name: str):
        """Initialize the base model.
        
        Args:
            config: Configuration dictionary.
            model_name: Name of the model (used for saving/loading).
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.feature_importances_ = None
        self.metrics = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        # Set random seeds for reproducibility
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)
        
        # Create output directories
        self.models_dir = Path(self.config['output']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'BaseModel':
        """Train the model on the given data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            
        Returns:
            The trained model instance.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the input data.
        
        Args:
            X: Input features.
            
        Returns:
            Array of predicted probabilities for the positive class.
        """
        pass
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels for the input data.
        
        Args:
            X: Input features.
            threshold: Classification threshold.
            
        Returns:
            Array of predicted class labels (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split: str = 'test') -> Dict[str, float]:
        """Evaluate the model on the given data.
        
        Args:
            X: Input features.
            y: True labels.
            split: Data split name (train/val/test).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'average_precision': average_precision_score(y, y_pred_proba)
        }
        
        # Update metrics dictionary
        self.metrics[split] = metrics
        
        # Log metrics
        logger.info(f"\n{self.model_name} {split} metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save the model to disk.
        
        Args:
            filename: Output filename (without path).
            
        Returns:
            Path to the saved model file.
        """
        if filename is None:
            filename = f"{self.model_name}.pkl"
        
        model_path = self.models_dir / filename
        joblib.dump(self, model_path)
        logger.info(f"Saved {self.model_name} model to {model_path}")
        return str(model_path)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model file.
            
        Returns:
            Loaded model instance.
        """
        return joblib.load(filepath)
    
    def get_feature_importances(self) -> pd.Series:
        """Get feature importances from the model.
        
        Returns:
            Series of feature importances.
        """
        if self.feature_importances_ is not None:
            return self.feature_importances_
        return pd.Series()


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: Dict):
        """Initialize the XGBoost model.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config, 'xgboost')
        
        # Get model parameters from config
        self.params = self.config['models']['xgb'].copy()
        self.params['random_state'] = self.seed
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'XGBoostModel':
        """Train the XGBoost model."""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=self.params.get('verbose', 100),
            early_stopping_rounds=self.params.get('early_stopping_rounds', None)
        )
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def __init__(self, config: Dict):
        """Initialize the LightGBM model."""
        super().__init__(config, 'lightgbm')
        
        # Get model parameters from config
        self.params = self.config['models']['lgb'].copy()
        self.params['random_state'] = self.seed
        
        # Initialize model
        self.model = lgb.LGBMClassifier(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'LightGBMModel':
        """Train the LightGBM model."""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val), (X_train, y_train)]
            eval_names = ['valid', 'train']
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names if eval_set else None,
            verbose=self.params.get('verbose', 100),
            early_stopping_rounds=self.params.get('early_stopping_rounds', None)
        )
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, config: Dict):
        """Initialize the Logistic Regression model."""
        super().__init__(config, 'logistic_regression')
        
        # Get model parameters from config
        self.params = self.config['models']['lr'].copy()
        self.params['random_state'] = self.seed
        
        # Initialize model
        self.model = LogisticRegression(**self.params)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'LogisticRegressionModel':
        """Train the Logistic Regression model."""
        self.model.fit(X_train, y_train)
        
        # For logistic regression, we can use absolute coefficients as importance
        self.feature_importances_ = pd.Series(
            np.abs(self.model.coef_[0]),
            index=X_train.columns
        ).sort_values(ascending=False)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]


def create_model(model_type: str, config: Dict) -> BaseModel:
    """Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create ('xgboost', 'lightgbm', or 'logistic_regression').
        config: Configuration dictionary.
        
    Returns:
        An instance of the specified model.
    """
    model_map = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(model_map.keys())}")
    
    return model_map[model_type](config)

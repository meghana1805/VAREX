""
Stacking ensemble implementation for MAGPIE-Lite.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import KFold
from .models import BaseModel
import logging

logger = logging.getLogger(__name__)

class StackingEnsemble:
    """A stacking ensemble model that combines multiple base models with a meta-learner."""
    
    def __init__(self, base_models: List[BaseModel], meta_model: BaseModel, 
                 n_splits: int = 5, random_state: int = 42):
        """Initialize the stacking ensemble.
        
        Args:
            base_models: List of base models to use in the ensemble.
            meta_model: Meta-learner model that will learn to combine base models.
            n_splits: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.feature_importances_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackingEnsemble':
        """Train the stacking ensemble.
        
        Args:
            X: Training features.
            y: Training labels.
            
        Returns:
            The trained stacking ensemble.
        """
        logger.info("Training stacking ensemble...")
        
        # Initialize out-of-fold predictions matrix
        n_samples = len(X)
        n_models = len(self.base_models)
        
        # Create array to store out-of-fold predictions
        X_meta = np.zeros((n_samples, n_models))
        
        # Dictionary to store trained base models
        self.trained_base_models_ = []
        
        # Train each base model and get out-of-fold predictions
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{n_models}: {model.model_name}")
            
            # Store trained models for each fold
            fold_models = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X, y), 1):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                
                # Train model on this fold
                model_copy = model.__class__(model.config)
                model_copy.fit(X_train_fold, y_train_fold)
                fold_models.append(model_copy)
                
                # Get predictions for validation set
                y_pred = model_copy.predict_proba(X_val_fold)
                X_meta[val_idx, i] = y_pred
                
                logger.debug(f"  Fold {fold_idx} - {model.model_name} trained")
            
            # Train final model on full training data for inference
            final_model = model.__class__(model.config)
            final_model.fit(X, y)
            
            # Store the final model and fold models
            self.trained_base_models_.append({
                'model': final_model,
                'fold_models': fold_models
            })
        
        # Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner...")
        self.meta_model_ = self.meta_model
        self.meta_model_.fit(pd.DataFrame(X_meta, columns=[f"model_{i}" for i in range(n_models)]), y)
        
        # Calculate feature importances from base models
        self._calculate_feature_importances(X.columns)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using the stacking ensemble.
        
        Args:
            X: Input features.
            
        Returns:
            Array of predicted probabilities for the positive class.
        """
        if not hasattr(self, 'trained_base_models_'):
            raise RuntimeError("Model not fitted. Call 'fit' before 'predict_proba'.")
        
        # Get predictions from each base model
        base_preds = []
        for model_dict in self.trained_base_models_:
            # Use the final model trained on full data for prediction
            y_pred = model_dict['model'].predict_proba(X)
            base_preds.append(y_pred)
        
        # Stack predictions horizontally
        X_meta = np.column_stack(base_preds)
        
        # Get final predictions from meta-learner
        return self.meta_model_.predict_proba(X_meta)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels using the stacking ensemble.
        
        Args:
            X: Input features.
            threshold: Classification threshold.
            
        Returns:
            Array of predicted class labels (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def _calculate_feature_importances(self, feature_names: List[str]) -> None:
        """Calculate feature importances by combining importances from base models.
        
        Args:
            feature_names: List of feature names.
        """
        # Initialize importance array
        importances = np.zeros(len(feature_names))
        
        # Sum importances from all base models
        for model_dict in self.trained_base_models_:
            model = model_dict['model']
            if hasattr(model, 'feature_importances_'):
                # Get feature importances from the model
                model_importances = model.get_feature_importances()
                # Ensure the feature order matches
                for i, feature in enumerate(feature_names):
                    if feature in model_importances:
                        importances[i] += model_importances[feature]
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    def get_feature_importances(self) -> pd.Series:
        """Get combined feature importances from the ensemble.
        
        Returns:
            Series of feature importances.
        """
        if self.feature_importances_ is None:
            raise RuntimeError("Feature importances not calculated. Call 'fit' first.")
        return self.feature_importances_


def create_stacking_ensemble(config: dict) -> StackingEnsemble:
    """Create a stacking ensemble from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        A configured StackingEnsemble instance.
    """
    from .models import create_model
    
    # Create base models
    base_models = [
        create_model('xgboost', config),
        create_model('lightgbm', config),
        create_model('logistic_regression', config)
    ]
    
    # Create meta-learner (Logistic Regression)
    meta_model = create_model('logistic_regression', config)
    
    # Create stacking ensemble
    return StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        n_splits=config['training']['n_splits'],
        random_state=config.get('seed', 42)
    )

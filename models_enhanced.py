"""
Enhanced model implementations for MAGPIE-Lite with additional functionality
like hyperparameter tuning and SHAP explainability.
"""
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, List
import joblib
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, train_test_split
)
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
from optuna.integration import OptunaSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models with common functionality."""
    
    def __init__(self, model_params: Dict[str, Any] = None):
        self.model = None
        self.model_params = model_params or {}
        self.feature_importances_ = None
        self.best_params_ = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.classes_ = None
    
    @abstractmethod
    def _init_model(self):
        """Initialize the model with parameters."""
        pass
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data."""
        if fit_scaler and y is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y.values
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the model on the given data."""
        if self.model is None:
            self._init_model()
        
        # Store feature names and classes
        self.feature_names = X.columns.tolist()
        self.classes_ = y.unique()
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True)
        
        logger.info(f"Training {self.__class__.__name__}...")
        self.model.fit(X_processed, y_processed, **kwargs)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, take absolute value of coefficients
            self.feature_importances_ = np.abs(self.model.coef_[0])
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_processed = self.preprocess_data(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_processed)
        else:
            # For models that don't support predict_proba, return hard predictions
            preds = self.predict(X_processed)
            proba = np.zeros((len(preds), 2))
            proba[:, 1] = preds
            proba[:, 0] = 1 - preds
            return proba
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metrics: List[str] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, float], Tuple]:
        """Evaluate the model on test data."""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1] if any(m in metrics for m in ['roc_auc', 'pr_auc']) else None
        
        results = {}
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == 'precision':
                results[metric] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                results[metric] = recall_score(y_test, y_pred)
            elif metric == 'f1':
                results[metric] = f1_score(y_test, y_pred)
            elif metric == 'roc_auc' and y_proba is not None:
                results[metric] = roc_auc_score(y_test, y_proba)
            elif metric == 'pr_auc' and y_proba is not None:
                results[metric] = average_precision_score(y_test, y_proba)
        
        if return_dict:
            return results
        return tuple(results.values())
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5,
        metrics: List[str] = None,
        n_jobs: int = -1,
        random_state: int = 42
    ) -> Dict[str, Tuple[float, float]]:
        """Perform cross-validation."""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        results = {}
        
        for metric in metrics:
            if metric == 'roc_auc':
                scorer = make_scorer(roc_auc_score, needs_proba=True)
            else:
                scorer = metric
                
            scores = cross_val_score(
                self.model, 
                self.scaler.transform(X), 
                y,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs
            )
            results[metric] = (scores.mean(), scores.std())
        
        return results
    
    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, Any],
        cv: int = 5,
        n_iter: int = 10,
        search_type: str = 'random',
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Perform hyperparameter tuning using grid or random search."""
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state
            )
        else:
            raise ValueError("search_type must be either 'grid' or 'random'")
        
        # Preprocess data
        X_processed = self.scaler.fit_transform(X)
        
        # Perform the search
        search.fit(X_processed, y)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': search.cv_results_
        }
    
    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict[str, Any],
        n_trials: int = 100,
        cv: int = 5,
        scoring: str = 'roc_auc',
        direction: str = 'maximize',
        n_jobs: int = 1,
        verbose: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for name, distribution in param_space.items():
                if distribution['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(
                        name, distribution['values']
                    )
                elif distribution['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, 
                        distribution['low'], 
                        distribution['high'],
                        step=distribution.get('step', 1)
                    )
                elif distribution['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, 
                        distribution['low'], 
                        distribution['high'],
                        log=distribution.get('log', False)
                    )
            
            # Create and train model with sampled parameters
            model = self.__class__(params)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                
                if scoring == 'roc_auc':
                    y_proba = model.predict_proba(X_val_fold)[:, 1]
                    score = roc_auc_score(y_val_fold, y_proba)
                elif scoring == 'accuracy':
                    y_pred = model.predict(X_val_fold)
                    score = accuracy_score(y_val_fold, y_pred)
                elif scoring == 'f1':
                    y_pred = model.predict(X_val_fold)
                    score = f1_score(y_val_fold, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")
                
                scores.append(score)
            
            # Return the mean score
            return np.mean(scores)
        
        # Run optimization
        study = optuna.create_study(direction=direction, sampler=TPESampler(seed=random_state))
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=n_jobs,
            show_progress_bar=verbose
        )
        
        # Update model with best parameters
        self.best_params_ = study.best_params
        self.model = self.__class__(study.best_params)
        self.model.fit(X, y)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def plot_feature_importance(
        self, 
        feature_names: List[str] = None,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """Plot feature importances."""
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet.")
            
        if feature_names is None:
            feature_names = self.feature_names or [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        # Sort features by importance
        indices = np.argsort(self.feature_importances_)[::-1]
        top_indices = indices[:top_n]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(
            range(len(top_indices)),
            self.feature_importances_[top_indices],
            align='center'
        )
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def explain_prediction(
        self, 
        X: pd.DataFrame,
        sample_idx: int = 0,
        plot_type: str = 'bar',
        show: bool = True,
        **shap_kwargs
    ) -> Union[shap.Explanation, plt.Figure]:
        """Explain model predictions using SHAP values."""
        if not hasattr(self, 'explainer'):
            X_processed = self.scaler.transform(X)
            self.explainer = shap.Explainer(self.model, X_processed, feature_names=self.feature_names)
        
        # Calculate SHAP values
        X_processed = self.scaler.transform(X)
        shap_values = self.explainer.shap_values(X_processed)
        
        # For binary classification, use the second class (positive class) values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Create the appropriate plot
        if plot_type == 'bar':
            fig = shap.plots.bar(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value,
                    data=X.iloc[sample_idx].values,
                    feature_names=self.feature_names
                ),
                show=show
            )
        elif plot_type == 'beeswarm':
            fig = shap.plots.beeswarm(
                shap.Explanation(
                    values=shap_values,
                    base_values=self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value,
                    data=X_processed,
                    feature_names=self.feature_names
                ),
                show=show,
                **shap_kwargs
            )
        elif plot_type == 'waterfall':
            fig = shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value,
                    data=X.iloc[sample_idx].values,
                    feature_names=self.feature_names
                ),
                show=show
            )
        elif plot_type == 'force':
            fig = shap.plots.force(
                self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value,
                shap_values[sample_idx],
                X.iloc[sample_idx],
                show=show,
                matplotlib=True
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        return fig
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load a model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


class XGBoostModel(BaseModel):
    """XGBoost classifier with enhanced functionality."""
    
    def _init_model(self):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**self.model_params)
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space for XGBoost."""
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
            'gamma': {'type': 'float', 'low': 0, 'high': 5},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 1, 'high': 10}
        }


class LightGBMModel(BaseModel):
    """LightGBM classifier with enhanced functionality."""
    
    def _init_model(self):
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(**self.model_params)
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space for LightGBM."""
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 100},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
        }


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier with enhanced functionality."""
    
    def _init_model(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(**self.model_params, max_iter=1000)
    
    def get_default_param_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space for Logistic Regression."""
        return {
            'C': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True},
            'penalty': {'type': 'categorical', 'values': ['l1', 'l2', 'elasticnet']},
            'solver': {'type': 'categorical', 'values': ['saga']},
            'l1_ratio': {'type': 'float', 'low': 0, 'high': 1}
        }
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data."""
        if fit_scaler and y is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if y is not None:
            return X_scaled, y.values
        return X_scaled

"""
Train Stacking Ensemble for MAGPIE-Lite.
"""
import os
import sys
import yaml
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model
from stacking import StackingEnsemble, create_stacking_ensemble
from data_utils import DataLoader, setup_logging

def load_config(config_path: str = '../config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_models(config: Dict[str, Any]) -> List[Any]:
    """Load pre-trained base models."""
    models_dir = Path(config['output']['models_dir'])
    model_files = list(models_dir.glob('*_model_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError("No trained models found. Please train base models first.")
    
    models = []
    for model_file in model_files:
        model = joblib.load(model_file)
        models.append(model)
        logging.info(f"Loaded model: {model_file.name}")
    
    return models

def train_stacking() -> None:
    """Train stacking ensemble model."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Stacking Ensemble training...")
    
    # Load configuration
    config = load_config()
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val = data_loader.get_train_val_split(
        test_size=config['training']['test_size'],
        random_state=config['seed']
    )
    
    # Create or load base models
    base_models = [
        create_model('xgboost', config),
        create_model('lightgbm', config),
        create_model('logistic_regression', config)
    ]
    
    # Create stacking ensemble
    logger.info("Creating stacking ensemble...")
    ensemble = create_stacking_ensemble(config)
    
    # Train ensemble
    logger.info("Training stacking ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating ensemble...")
    val_preds = ensemble.predict_proba(X_val)
    
    # Save ensemble
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_path = os.path.join(config['output']['models_dir'], f'stacking_ensemble_{timestamp}.pkl')
    joblib.dump(ensemble, ensemble_path)
    logger.info(f"Stacking ensemble saved to {ensemble_path}")
    
    # Save feature importances
    importances = ensemble.get_feature_importances()
    importances_path = os.path.join(config['output']['models_dir'], 'stacking_feature_importances.csv')
    importances.to_csv(importances_path)
    logger.info(f"Feature importances saved to {importances_path}")
    
    logger.info("Stacking ensemble training completed successfully!")

if __name__ == "__main__":
    train_stacking()

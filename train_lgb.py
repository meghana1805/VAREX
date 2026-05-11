"""
Train LightGBM model for MAGPIE-Lite.
"""
import os
import sys
import yaml
import joblib
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import LightGBMModel
from data_utils import DataLoader, setup_logging

def load_config(config_path: str = '../config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_lightgbm() -> None:
    """Train LightGBM model with cross-validation."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting LightGBM training...")
    
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
    
    # Train model
    logger.info("Training LightGBM model...")
    model = LightGBMModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    val_metrics = model.evaluate(X_val, y_val, 'val')
    
    # Save model
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config['output']['models_dir'], f'lgb_model_{timestamp}.pkl')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature importances
    importances = model.get_feature_importances()
    importances_path = os.path.join(config['output']['models_dir'], 'lgb_feature_importances.csv')
    importances.to_csv(importances_path)
    logger.info(f"Feature importances saved to {importances_path}")
    
    logger.info("LightGBM training completed successfully!")

if __name__ == "__main__":
    train_lightgbm()

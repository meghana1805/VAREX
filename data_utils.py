"""
Data loading and preprocessing utilities for MAGPIE-Lite.
"""
import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import numpy as np
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of variant data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the DataLoader with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self.raw_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self.required_features = self.config['features']['required_features']
        self.target = self.config['features']['target']
        self.missing_value = self.config['features']['missing_value']
        
        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load (train, test, orthogonal, acmg, denovo).
            
        Returns:
            Loaded DataFrame with required features.
        """
        file_map = {
            'train': self.config['data']['train_file'],
            'test': self.config['data']['test_file'],
            'orthogonal': self.config['data']['orthogonal_file'],
            'acmg': self.config['data']['acmg_file'],
            'denovo': self.config['data']['denovo_file']
        }
        
        if dataset_name not in file_map:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(file_map.keys())}")
        
        file_path = self.raw_path / file_map[dataset_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading {dataset_name} dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Ensure required features exist
        self._validate_features(df, dataset_name)
        
        return df
    
    def _validate_features(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate that required features exist in the dataset.
        
        Args:
            df: Input DataFrame to validate.
            dataset_name: Name of the dataset for logging purposes.
        """
        missing_features = [f for f in self.required_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in {dataset_name}: {missing_features}")
            
            # Add missing features with sentinel value
            for feature in missing_features:
                if feature != self.target:  # Don't add target if missing
                    df[feature] = self.missing_value
                    logger.warning(f"Added missing feature '{feature}' with value {self.missing_value}")
        
        # Ensure target exists for training data
        if dataset_name == 'train' and self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in training data")
    
    def preprocess_data(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the input DataFrame.
        
        Args:
            df: Input DataFrame.
            is_train: Whether this is training data (needs target).
            
        Returns:
            Tuple of (features, target) if is_train else (features, None)
        """
        # Ensure all required features are present
        features = [f for f in self.required_features if f in df.columns]
        
        # Handle missing values
        X = df[features].copy()
        X = X.replace([np.inf, -np.inf], self.missing_value)
        
        # For training data, also return the target
        if is_train:
            if self.target not in df.columns:
                raise ValueError(f"Target column '{self.target}' not found in training data")
            y = df[self.target].copy()
            return X, y
        
        return X, None
    
    def get_train_val_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and split the training data into train/validation sets.
        
        Args:
            test_size: Fraction of data to use for validation.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        df = self.load_dataset('train')
        X, y = self.preprocess_data(df, is_train=True)
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to disk.
        
        Args:
            df: DataFrame to save.
            filename: Output filename (without path).
        """
        output_path = self.processed_path / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the configuration dictionary.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up basic logging configuration.
    
    Args:
        log_level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('magpie_lite.log')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")

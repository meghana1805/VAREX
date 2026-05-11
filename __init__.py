"""
MAGPIE-Lite - A lightweight variant pathogenicity prediction framework
"""

# Import key classes for easier access
from .data_utils import DataLoader
from .models_enhanced import (
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    LogisticRegressionModel
)

__version__ = '0.1.0'
__author__ = 'Your Name <your.email@example.com>'
__all__ = [
    'DataLoader',
    'BaseModel',
    'XGBoostModel',
    'LightGBMModel',
    'LogisticRegressionModel'
]

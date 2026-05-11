"""
Model evaluation script for MAGPIE-Lite.
"""
import os
import sys
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_utils import DataLoader, setup_logging

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class ModelEvaluator:
    """Class for evaluating model performance."""
    
    def __init__(self, config_path: str = '../config.yaml'):
        """Initialize the evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.output_dir = Path(self.config['output']['figures_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_path: Union[str, Path]):
        """Load a trained model from disk."""
        return joblib.load(model_path)
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'test') -> Dict[str, float]:
        """Evaluate a model and return metrics."""
        # Get predictions
        y_pred_proba = model.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'average_precision': average_precision_score(y, y_pred_proba)
        }
        
        # Log metrics
        self.logger.info(f"\nMetrics for {dataset_name}:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics, y_pred_proba
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str, dataset_name: str) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} ({dataset_name})')
        plt.legend(loc="lower right")
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_roc_{dataset_name}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                     model_name: str, dataset_name: str) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure()
        plt.step(recall, precision, where='post', 
                label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {model_name} ({dataset_name})')
        plt.legend(loc='lower left')
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_pr_{dataset_name}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str, dataset_name: str) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=['Benign', 'Pathogenic'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
        
        # Save figure
        filename = f"{model_name.lower().replace(' ', '_')}_cm_{dataset_name}.png"
        plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                              model_name: str) -> None:
        """Plot feature importances."""
        if hasattr(model, 'get_feature_importances'):
            importances = model.get_feature_importances()
            
            # Create a DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Feature Importances - {model_name}')
            plt.tight_layout()
            
            # Save figure
            filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(self.output_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
    
    def evaluate_on_datasets(self, model, model_name: str, 
                           datasets: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """Evaluate model on multiple datasets."""
        results = {}
        
        for dataset_name, dataset_path in datasets:
            self.logger.info(f"\nEvaluating on {dataset_name}...")
            
            # Load and preprocess data
            df = self.data_loader.load_dataset(dataset_name)
            X, y = self.data_loader.preprocess_data(df, is_train=True)
            
            # Evaluate
            metrics, y_pred_proba = self.evaluate_model(model, X, y, dataset_name)
            results[dataset_name] = metrics
            
            # Generate plots
            y_pred = (y_pred_proba >= 0.5).astype(int)
            self.plot_roc_curve(y, y_pred_proba, model_name, dataset_name)
            self.plot_pr_curve(y, y_pred_proba, model_name, dataset_name)
            self.plot_confusion_matrix(y, y_pred, model_name, dataset_name)
            
            # Plot feature importance if available
            if dataset_name == 'train':  # Only plot feature importance once
                self.plot_feature_importance(model, X.columns.tolist(), model_name)
        
        return results

def main():
    """Main evaluation function."""
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load configuration
    config = evaluator.config
    models_dir = Path(config['output']['models_dir'])
    
    # Define datasets to evaluate on
    datasets = [
        ('train', 'train'),
        ('test', 'test'),
        ('orthogonal', 'orthogonal'),
        ('acmg', 'acmg_guided'),
        ('denovo', 'denovo')
    ]
    
    # Find all model files
    model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        evaluator.logger.error("No model files found. Please train models first.")
        return
    
    # Evaluate each model
    all_results = {}
    for model_file in model_files:
        model_name = model_file.stem
        evaluator.logger.info(f"\n{'='*50}")
        evaluator.logger.info(f"Evaluating model: {model_name}")
        evaluator.logger.info("="*50)
        
        try:
            # Load model
            model = evaluator.load_model(model_file)
            
            # Evaluate on all datasets
            results = evaluator.evaluate_on_datasets(
                model, model_name, datasets
            )
            all_results[model_name] = results
            
            # Save results to CSV
            results_df = pd.DataFrame(results).T
            results_path = models_dir / f"{model_name}_results.csv"
            results_df.to_csv(results_path)
            evaluator.logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            evaluator.logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    # Compare all models
    if len(all_results) > 1:
        compare_models(all_results, evaluator.output_dir)
    
    evaluator.logger.info("\nEvaluation completed successfully!")

def compare_models(all_results: Dict[str, Dict[str, Dict[str, float]]], 
                  output_dir: Path) -> None:
    """Generate comparison plots for all models."""
    # Convert results to DataFrame for easier plotting
    comparison_data = []
    for model_name, results in all_results.items():
        for dataset_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                **metrics
            })
    
    if not comparison_data:
        return
    
    df = pd.DataFrame(comparison_data)
    
    # Plot comparison for each metric
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Dataset', y=metric, hue='Model', data=df)
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save figure
            filename = f"model_comparison_{metric}.png"
            plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    main()

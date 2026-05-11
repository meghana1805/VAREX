#!/usr/bin/env python3
"""
MAGPIE-Lite Pipeline Runner

This script provides a command-line interface to run the entire MAGPIE-Lite pipeline
or individual components as needed.
"""
import os
import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import models after path is set
try:
    from src.models_enhanced import XGBoostModel, LightGBMModel, LogisticRegressionModel
    from src.data_utils import DataLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you've installed all requirements and the project structure is correct.")
    sys.exit(1)

class PipelineRunner:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            return {}
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            'data/processed',
            'models',
            'reports/figures',
            'reports/metrics'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def run_data_processing(self):
        """Run data processing pipeline."""
        print("\n" + "="*50)
        print("Starting data processing...")
        try:
            # Add your data processing logic here
            print("Data processing completed successfully.")
            return True
        except Exception as e:
            print(f"Error during data processing: {e}")
            return False
    
    def train_models(self, model_type: str = 'all'):
        """Train machine learning models."""
        print("\n" + "="*50)
        print("Starting model training...")
        
        models = {
            'xgb': XGBoostModel(),
            'lgb': LightGBMModel(),
            'lr': LogisticRegressionModel()
        }
        
        if model_type != 'all':
            models = {k: v for k, v in models.items() if k == model_type}
        
        results = {}
        for name, model in models.items():
            try:
                print(f"\nTraining {name.upper()} model...")
                
                # Load and preprocess data
                X, y = self.data_loader.load_training_data()
                
                # Train model
                model.fit(X, y)
                
                # Save model
                model_path = f"models/{name}_model.pkl"
                model.save(model_path)
                print(f"Model saved to {model_path}")
                results[name] = "Success"
            except Exception as e:
                print(f"Error training {name}: {e}")
                results[name] = f"Failed: {e}"
        
        return results
    
    def evaluate_models(self):
        """Evaluate trained models."""
        print("\n" + "="*50)
        print("Evaluating models...")
        # Add your evaluation logic here
        print("Evaluation completed.")
        return {}
    
    def make_predictions(self, input_file: str, output_file: str = None):
        """Make predictions using trained models."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        try:
            print(f"\nMaking predictions on {input_file}...")
            
            # Load data
            data = pd.read_csv(input_file)
            
            # Load model (using XGBoost as default)
            model_path = "models/xgb_model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
            
            model = XGBoostModel.load(model_path)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Save predictions
            output_file = output_file or f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error making predictions: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='MAGPIE-Lite Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data processing command
    process_parser = subparsers.add_parser('process', help='Process raw data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', type=str, default='all',
                            choices=['all', 'xgb', 'lgb', 'lr'],
                            help='Model to train')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('input_file', type=str, help='Input CSV file')
    predict_parser.add_argument('--output', type=str, help='Output file path')
    
    # Run all command
    all_parser = subparsers.add_parser('run_all', help='Run entire pipeline')
    
    args = parser.parse_args()
    
    try:
        pipeline = PipelineRunner()
        
        if args.command == 'process':
            success = pipeline.run_data_processing()
            sys.exit(0 if success else 1)
        elif args.command == 'train':
            results = pipeline.train_models(args.model)
            print("\nTraining Summary:")
            for model, status in results.items():
                print(f"- {model.upper()}: {status}")
        elif args.command == 'evaluate':
            pipeline.evaluate_models()
        elif args.command == 'predict':
            success = pipeline.make_predictions(args.input_file, args.output)
            sys.exit(0 if success else 1)
        elif args.command == 'run_all':
            print("="*50)
            print("Running MAGPIE-Lite Pipeline")
            print("="*50)
            
            # Run data processing
            if not pipeline.run_data_processing():
                print("\nData processing failed. Exiting.")
                sys.exit(1)
                
            # Train models
            results = pipeline.train_models()
            print("\nTraining Summary:")
            for model, status in results.items():
                print(f"- {model.upper()}: {status}")
                
            # Evaluate models
            pipeline.evaluate_models()
            
            print("\n" + "="*50)
            print("Pipeline completed successfully!")
            print("="*50)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# MAGPIE-Lite: Lightweight Pathogenicity Prediction Framework

A high-performance, interpretable variant pathogenicity prediction framework implementing a stacked ensemble model with XGBoost, LightGBM, and Logistic Regression.

## Features

- 🚀 Optimized for speed with only 7-10 high-value biological features
- 📊 Stacked ensemble model with meta-learning
- 🔍 Explainable AI with SHAP values
- 📈 Reproducible research with configuration management
- 🐳 Docker support for easy deployment
- 📊 Paper-ready visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/magpie-lite.git
cd magpie-lite

# Create and activate conda environment
conda create -n magpie-lite python=3.10
conda activate magpie-lite

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
magpie-lite/
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   └── processed/         # Processed datasets
├── notebooks/             # Jupyter notebooks for EDA and analysis
├── scripts/               # Main pipeline scripts
├── src/                   # Source code modules
├── tests/                 # Unit tests
├── artifacts/             # Model artifacts and outputs
│   ├── models/            # Saved models
│   ├── figures/           # Generated plots
│   └── reports/           # Generated reports
├── config.yaml            # Configuration file
├── Dockerfile             # Container definition
└── requirements.txt       # Python dependencies
```

## Quick Start

1. Place your input CSV files in `data/raw/`
2. Run the preprocessing pipeline:
   ```bash
   python scripts/preprocess.py
   ```
3. Train the base models:
   ```bash
   python scripts/train_xgb.py
   python scripts/train_lgb.py
   python scripts/train_lr.py
   ```
4. Train the stacked ensemble:
   ```bash
   python scripts/stack_train.py
   ```
5. Generate predictions:
   ```bash
   python scripts/predict.py
   ```
6. Generate visualizations:
   ```bash
   python scripts/shap_explain.py
   ```

## Model Architecture

MAGPIE-Lite uses a stacked ensemble approach:

1. **Base Models**:
   - XGBoost
   - LightGBM
   - Logistic Regression

2. **Meta-learner**:
   - Logistic Regression trained on out-of-fold predictions



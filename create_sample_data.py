import pandas as pd
import numpy as np
from pathlib import Path

# Create directories if they don't exist
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Sample data parameters
n_samples = 1000
n_features = 20

# Generate random features
np.random.seed(42)
X = np.random.randn(n_samples, n_features)

# Generate target variable (binary classification)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5) > 0
y = y.astype(int)

# Create feature names
feature_columns = [f"feature_{i+1}" for i in range(n_features)]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_columns)
df['target'] = y

# Add some missing values (10% of each column)
for col in df.columns:
    mask = np.random.rand(len(df)) < 0.1
    df.loc[mask, col] = np.nan

# Save to CSV
output_path = data_dir / "train.csv"
df.to_csv(output_path, index=False)
print(f"Sample dataset created at: {output_path}")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

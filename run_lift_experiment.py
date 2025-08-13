#!/usr/bin/env python3
"""
Simple LIFT Experiment Runner

This is a basic example script showing how to use the LIFT package
for multilabel classification on one of the available datasets.

For a more comprehensive system, use:
- multilabel_trainer.py for interactive training
- lift_inference.py for model inference
- dataset_explorer.py for dataset analysis
- batch_runner.py for batch experiments

Author: GitHub Copilot
"""

import sys
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss, jaccard_score

# Add LIFT package to path
sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')

try:
    from lift_ml import LIFTClassifier
except ImportError:
    print("❌ LIFT package not found!")
    print("Please run: python setup.py")
    sys.exit(1)


def extract_dataset(dataset_name):
    """Extract a dataset if not already extracted."""
    zip_path = Path(f"{dataset_name}.zip")
    extract_path = Path(f"extracted_datasets/{dataset_name}")
    
    if not zip_path.exists():
        print(f"❌ Dataset {dataset_name}.zip not found!")
        return None
    
    if extract_path.exists():
        print(f"✅ Dataset {dataset_name} already extracted")
        return extract_path
    
    print(f"📦 Extracting {dataset_name}...")
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    return extract_path


def load_simple_dataset(dataset_path):
    """Load the first CSV file found in the dataset."""
    csv_files = list(dataset_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {dataset_path}")
        return None, None, None, None
    
    csv_file = csv_files[0]
    print(f"📊 Loading {csv_file.name}...")
    
    df = pd.read_csv(csv_file)
    print(f"   Shape: {df.shape}")
    
    # Simple heuristic: assume last 5 columns are labels, rest are features
    n_labels = 5
    feature_cols = df.columns[:-n_labels].tolist()
    label_cols = df.columns[-n_labels:].tolist()
    
    # Verify labels are binary
    for col in label_cols:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            # Adjust if not all are binary
            n_labels = len([c for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})])
            feature_cols = df.columns[:-n_labels].tolist() if n_labels > 0 else df.columns[:-3].tolist()
            label_cols = df.columns[-n_labels:].tolist() if n_labels > 0 else df.columns[-3:].tolist()
            break
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Labels: {len(label_cols)}")
    
    X = df[feature_cols].values
    y = df[label_cols].values
    
    return X, y, feature_cols, label_cols


def run_simple_experiment(dataset_name="yeast"):
    """Run a simple LIFT experiment on a dataset."""
    print(f"🚀 Running LIFT experiment on {dataset_name}")
    print("="*50)
    
    # Extract dataset
    dataset_path = extract_dataset(dataset_name)
    if dataset_path is None:
        return
    
    # Load data
    X, y, feature_names, label_names = load_simple_dataset(dataset_path)
    if X is None:
        return
    
    # Train/test split
    print("🔄 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Scale features
    print("🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train LIFT model
    print("🤖 Training LIFT classifier...")
    clf = LIFTClassifier(
        k=3,  # Number of clusters
        random_state=42
    )
    
    clf.fit(X_train_scaled, y_train)
    print("✅ Training completed!")
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate
    print("📊 Evaluating performance...")
    metrics = {
        'Hamming Loss': hamming_loss(y_test, y_pred),
        'Jaccard Score (micro)': jaccard_score(y_test, y_pred, average='micro'),
        'Jaccard Score (macro)': jaccard_score(y_test, y_pred, average='macro'),
        'F1 Score (micro)': f1_score(y_test, y_pred, average='micro'),
        'F1 Score (macro)': f1_score(y_test, y_pred, average='macro'),
        'F1 Score (weighted)': f1_score(y_test, y_pred, average='weighted'),
    }
    
    print("\n📈 Results:")
    print("-"*30)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Per-label F1 scores
    print(f"\n🏷️  Per-label F1 scores:")
    for i, label in enumerate(label_names):
        f1 = f1_score(y_test[:, i], y_pred[:, i])
        print(f"   {label:15s}: {f1:.4f}")
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"💡 For more advanced features, try:")
    print(f"   python multilabel_trainer.py --interactive")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple LIFT Experiment")
    parser.add_argument("--dataset", "-d", default="yeast", 
                       help="Dataset name (default: yeast)")
    
    args = parser.parse_args()
    
    run_simple_experiment(args.dataset)
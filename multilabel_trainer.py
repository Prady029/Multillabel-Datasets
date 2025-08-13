#!/usr/bin/env python3
"""
Multilabel Dataset Training System using LIFT

This script provides an interactive interface for:
1. Dataset selection and preprocessing
2. LIFT model training with hyperparameter tuning
3. Model evaluation and reporting
4. Model persistence and inference

Author: GitHub Copilot
"""

import os
import sys
import json
import pickle
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, multilabel_confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add LIFT package to path
sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')
from lift_ml import LIFTClassifier


class MultilabelTrainer:
    """Main class for handling multilabel dataset training with LIFT."""
    
    def __init__(self):
        self.datasets_info = {
            'birds': {
                'description': 'Bird species classification from audio features',
                'file_pattern': 'birds-train.csv',
                'test_file': 'birds-test.csv'
            },
            'bookmarks': {
                'description': 'Web bookmark categorization',
                'file_pattern': 'bookmarks-train.csv',
                'test_file': 'bookmarks-test.csv'
            },
            'Cal500': {
                'description': 'Music emotion classification (CAL500)',
                'file_pattern': 'Cal500-train.csv',
                'test_file': 'Cal500-test.csv'
            },
            'corel5k': {
                'description': 'Image annotation with Corel 5K dataset',
                'file_pattern': 'corel5k-train.csv',
                'test_file': 'corel5k-test.csv'
            },
            'delicious': {
                'description': 'Social bookmarking tag prediction',
                'file_pattern': 'delicious-train.csv',
                'test_file': 'delicious-test.csv'
            },
            'Emotions': {
                'description': 'Music emotion classification',
                'file_pattern': 'emotions-train.csv',
                'test_file': 'emotions-test.csv'
            },
            'enron': {
                'description': 'Email classification (Enron dataset)',
                'file_pattern': 'enron-train.csv',
                'test_file': 'enron-test.csv'
            },
            'genbase': {
                'description': 'Gene functional classification',
                'file_pattern': 'genbase-train.csv',
                'test_file': 'genbase-test.csv'
            },
            'mediamill': {
                'description': 'Video semantic annotation',
                'file_pattern': 'mediamill-train.csv',
                'test_file': 'mediamill-test.csv'
            },
            'yeast': {
                'description': 'Yeast protein functional classification',
                'file_pattern': 'yeast-train.csv',
                'test_file': 'yeast-test.csv'
            }
        }
        
        self.workspace_dir = Path('.')
        self.data_dir = self.workspace_dir / 'extracted_datasets'
        self.models_dir = self.workspace_dir / 'trained_models'
        self.reports_dir = self.workspace_dir / 'reports'
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def list_available_datasets(self) -> None:
        """Display available datasets with descriptions."""
        print("\\n" + "="*60)
        print("AVAILABLE MULTILABEL DATASETS")
        print("="*60)
        
        for i, (dataset_name, info) in enumerate(self.datasets_info.items(), 1):
            zip_path = self.workspace_dir / f"{dataset_name}.zip"
            status = "✅ Available" if zip_path.exists() else "❌ Missing"
            print(f"{i:2d}. {dataset_name:<12} - {info['description']}")
            print(f"    Status: {status}")
        print("="*60)
    
    def extract_dataset(self, dataset_name: str) -> bool:
        """Extract dataset if not already extracted."""
        zip_path = self.workspace_dir / f"{dataset_name}.zip"
        extract_path = self.data_dir / dataset_name
        
        if not zip_path.exists():
            print(f"❌ Dataset {dataset_name} not found!")
            return False
        
        if extract_path.exists():
            print(f"✅ Dataset {dataset_name} already extracted.")
            return True
        
        try:
            print(f"📦 Extracting {dataset_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✅ Successfully extracted {dataset_name}")
            return True
        except Exception as e:
            print(f"❌ Error extracting {dataset_name}: {e}")
            return False
    
    def load_dataset(self, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                      Optional[np.ndarray], Optional[np.ndarray],
                                                      Optional[List[str]], Optional[List[str]]]:
        """Load and preprocess the dataset."""
        extract_path = self.data_dir / dataset_name
        
        # Find the actual data files
        train_files = list(extract_path.glob("*train*.csv")) + list(extract_path.glob("*-train.csv"))
        test_files = list(extract_path.glob("*test*.csv")) + list(extract_path.glob("*-test.csv"))
        
        if not train_files:
            # Look for any CSV files
            csv_files = list(extract_path.rglob("*.csv"))
            if csv_files:
                print(f"Found CSV files: {[f.name for f in csv_files]}")
                train_file = csv_files[0]  # Use the first one found
                test_file = None
            else:
                print(f"❌ No CSV files found in {extract_path}")
                return None, None, None, None, None, None
        else:
            train_file = train_files[0]
            test_file = test_files[0] if test_files else None
        
        try:
            print(f"📊 Loading training data from {train_file.name}...")
            train_df = pd.read_csv(train_file)
            
            # Auto-detect features and labels
            # Assume binary columns (0/1) at the end are labels
            binary_cols = []
            for col in train_df.columns:
                if set(train_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(col)
            
            # Heuristic: if more than half the columns are binary, assume they're labels
            if len(binary_cols) > len(train_df.columns) // 2:
                label_cols = binary_cols
                feature_cols = [col for col in train_df.columns if col not in label_cols]
            else:
                # Alternative: assume last few columns are labels
                n_features = len(train_df.columns) - len(binary_cols) if binary_cols else len(train_df.columns) - 5
                feature_cols = train_df.columns[:n_features].tolist()
                label_cols = train_df.columns[n_features:].tolist()
            
            print(f"   Features: {len(feature_cols)} columns")
            print(f"   Labels: {len(label_cols)} columns")
            print(f"   Samples: {len(train_df)}")
            
            X_train = train_df[feature_cols].values
            y_train = train_df[label_cols].values
            
            # Handle test set
            if test_file and test_file.exists():
                print(f"📊 Loading test data from {test_file.name}...")
                test_df = pd.read_csv(test_file)
                X_test = test_df[feature_cols].values
                y_test = test_df[label_cols].values
            else:
                print("📊 No separate test file found. Will use train/validation split.")
                X_test, y_test = None, None
            
            return X_train, X_test, y_train, y_test, feature_cols, label_cols
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {e}")
            return None, None, None, None, None, None
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   dataset_name: str, use_optimization: bool = True) -> LIFTClassifier:
        """Train LIFT model with optional hyperparameter optimization."""
        print(f"\\n🔧 Training LIFT model for {dataset_name}...")
        
        if use_optimization:
            print("⚡ Using Bayesian optimization for hyperparameter tuning...")
            from skopt.space import Integer
            
            search_space = {
                'lift__k': Integer(1, min(10, y_train.shape[0] // 10)),  # Adaptive k range
            }
            
            clf = LIFTClassifier(
                auto_tune=True,
                tune_params=search_space,
                n_iter=15,
                cv=3,
                random_state=42
            )
        else:
            print("🔧 Using default hyperparameters...")
            clf = LIFTClassifier(k=3, random_state=42)
        
        # Fit the model
        clf.fit(X_train, y_train)
        
        if use_optimization and hasattr(clf, 'best_params_'):
            print(f"✅ Best parameters found: {clf.best_params_}")
            print(f"✅ Best CV score: {clf.best_score_:.4f}")
        
        return clf
    
    def evaluate_model(self, clf: LIFTClassifier, X_test: np.ndarray, y_test: np.ndarray,
                      dataset_name: str, label_names: List[str]) -> Dict:
        """Comprehensive model evaluation."""
        print(f"\\n📊 Evaluating model performance...")
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(y_test),
            'n_labels': y_test.shape[1],
            'metrics': {
                'hamming_loss': hamming_loss(y_test, y_pred),
                'jaccard_score_micro': jaccard_score(y_test, y_pred, average='micro'),
                'jaccard_score_macro': jaccard_score(y_test, y_pred, average='macro'),
                'f1_score_micro': f1_score(y_test, y_pred, average='micro'),
                'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_score_weighted': f1_score(y_test, y_pred, average='weighted'),
                'subset_accuracy': accuracy_score(y_test, y_pred)
            },
            'per_label_metrics': {}
        }
        
        # Per-label metrics
        for i, label in enumerate(label_names):
            metrics['per_label_metrics'][label] = {
                'f1_score': f1_score(y_test[:, i], y_pred[:, i]),
                'precision': classification_report(y_test[:, i], y_pred[:, i], output_dict=True)['1']['precision'] if '1' in classification_report(y_test[:, i], y_pred[:, i], output_dict=True) else 0,
                'recall': classification_report(y_test[:, i], y_pred[:, i], output_dict=True)['1']['recall'] if '1' in classification_report(y_test[:, i], y_pred[:, i], output_dict=True) else 0,
                'support': int(y_test[:, i].sum())
            }
        
        # Print summary
        print("\\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Dataset: {dataset_name}")
        print(f"Test samples: {metrics['test_samples']}")
        print(f"Number of labels: {metrics['n_labels']}")
        print("\\nOverall Metrics:")
        for metric_name, value in metrics['metrics'].items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model_and_report(self, clf: LIFTClassifier, metrics: Dict, 
                             dataset_name: str, feature_names: List[str], 
                             label_names: List[str]) -> None:
        """Save trained model and generate comprehensive report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / f"{dataset_name}_lift_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': clf,
                'feature_names': feature_names,
                'label_names': label_names,
                'dataset_name': dataset_name,
                'timestamp': timestamp
            }, f)
        print(f"💾 Model saved to: {model_path}")
        
        # Save metrics report
        report_path = self.reports_dir / f"{dataset_name}_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📄 Report saved to: {report_path}")
        
        # Generate HTML report
        self.generate_html_report(metrics, dataset_name, timestamp)
    
    def generate_html_report(self, metrics: Dict, dataset_name: str, timestamp: str) -> None:
        """Generate a comprehensive HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LIFT Model Report - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LIFT Multilabel Classification Report</h1>
                <p><strong>Dataset:</strong> {dataset_name}</p>
                <p><strong>Timestamp:</strong> {metrics['timestamp']}</p>
                <p><strong>Test Samples:</strong> {metrics['test_samples']}</p>
                <p><strong>Number of Labels:</strong> {metrics['n_labels']}</p>
            </div>
            
            <div class="metrics">
                <h2>Overall Performance Metrics</h2>
        """
        
        for metric_name, value in metrics['metrics'].items():
            html_content += f'<div class="metric"><strong>{metric_name}:</strong> {value:.4f}</div>\\n'
        
        html_content += """
                <h2>Per-Label Performance</h2>
                <table>
                    <tr>
                        <th>Label</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Support</th>
                    </tr>
        """
        
        for label, label_metrics in metrics['per_label_metrics'].items():
            html_content += f"""
                    <tr>
                        <td>{label}</td>
                        <td>{label_metrics['f1_score']:.4f}</td>
                        <td>{label_metrics['precision']:.4f}</td>
                        <td>{label_metrics['recall']:.4f}</td>
                        <td>{label_metrics['support']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        report_path = self.reports_dir / f"{dataset_name}_report_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"🌐 HTML report saved to: {report_path}")
    
    def run_interactive_training(self) -> None:
        """Main interactive training workflow."""
        print("\\n🚀 LIFT Multilabel Training System")
        print("="*50)
        
        # Show available datasets
        self.list_available_datasets()
        
        # Get user choice
        while True:
            try:
                choice = input("\\n🔢 Select dataset number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    return
                
                dataset_idx = int(choice) - 1
                dataset_names = list(self.datasets_info.keys())
                
                if 0 <= dataset_idx < len(dataset_names):
                    dataset_name = dataset_names[dataset_idx]
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        print(f"\\n✅ Selected dataset: {dataset_name}")
        
        # Extract dataset
        if not self.extract_dataset(dataset_name):
            return
        
        # Load dataset
        X_train, X_test, y_train, y_test, feature_names, label_names = self.load_dataset(dataset_name)
        if X_train is None:
            return
        
        # Handle train/test split if no separate test set
        if X_test is None:
            print("📊 Creating train/validation split (80/20)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=None
            )
        
        # Ask about hyperparameter optimization
        use_optimization = input("\\n⚡ Use Bayesian optimization for hyperparameters? (y/n) [y]: ").strip().lower()
        use_optimization = use_optimization != 'n'
        
        # Scale features
        print("🔧 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        clf = self.train_model(X_train_scaled, y_train, dataset_name, use_optimization)
        
        # Evaluate model
        metrics = self.evaluate_model(clf, X_test_scaled, y_test, dataset_name, label_names)
        
        # Save everything
        self.save_model_and_report(clf, metrics, dataset_name, feature_names, label_names)
        
        print(f"\\n🎉 Training completed successfully for {dataset_name}!")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="LIFT Multilabel Training System")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--dataset", "-d", type=str,
                       help="Dataset name to train on")
    parser.add_argument("--optimize", "-o", action="store_true",
                       help="Use Bayesian optimization")
    
    args = parser.parse_args()
    
    trainer = MultilabelTrainer()
    
    if args.interactive or not args.dataset:
        trainer.run_interactive_training()
    else:
        # Non-interactive mode
        if args.dataset not in trainer.datasets_info:
            print(f"❌ Unknown dataset: {args.dataset}")
            trainer.list_available_datasets()
            return
        
        print(f"🚀 Training {args.dataset} in batch mode...")
        
        # Extract and load dataset
        if not trainer.extract_dataset(args.dataset):
            return
        
        X_train, X_test, y_train, y_test, feature_names, label_names = trainer.load_dataset(args.dataset)
        if X_train is None:
            return
        
        # Handle train/test split if needed
        if X_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate
        clf = trainer.train_model(X_train_scaled, y_train, args.dataset, args.optimize)
        metrics = trainer.evaluate_model(clf, X_test_scaled, y_test, args.dataset, label_names)
        trainer.save_model_and_report(clf, metrics, args.dataset, feature_names, label_names)
        
        print(f"🎉 Batch training completed for {args.dataset}!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LIFT Model Inference Script

This script provides functionality to:
1. Load trained LIFT models
2. Make predictions on new data
3. Batch inference on datasets
4. Generate prediction reports

Author: GitHub Copilot
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# Add LIFT package to path
sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')


class LIFTInference:
    """Class for making inferences with trained LIFT models."""
    
    def __init__(self):
        self.models_dir = Path('./trained_models')
        self.reports_dir = Path('./reports')
        self.predictions_dir = Path('./predictions')
        
        # Create directories if they don't exist
        self.predictions_dir.mkdir(exist_ok=True)
    
    def list_available_models(self) -> List[Dict]:
        """List all available trained models."""
        if not self.models_dir.exists():
            print("❌ No trained models directory found!")
            return []
        
        models = []
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                models.append({
                    'file': model_file.name,
                    'path': model_file,
                    'dataset': model_data.get('dataset_name', 'Unknown'),
                    'timestamp': model_data.get('timestamp', 'Unknown'),
                    'feature_count': len(model_data.get('feature_names', [])),
                    'label_count': len(model_data.get('label_names', []))
                })
            except Exception as e:
                print(f"⚠️  Could not load model {model_file.name}: {e}")
        
        return models
    
    def load_model(self, model_path: Path) -> Dict:
        """Load a trained model with all its metadata."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"✅ Loaded model for dataset: {model_data['dataset_name']}")
            print(f"   Features: {len(model_data['feature_names'])}")
            print(f"   Labels: {len(model_data['label_names'])}")
            print(f"   Trained: {model_data['timestamp']}")
            
            return model_data
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def predict_from_csv(self, model_data: Dict, csv_path: Path, 
                        output_path: Optional[Path] = None) -> np.ndarray:
        """Make predictions on data from a CSV file."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"📊 Loaded {len(df)} samples from {csv_path.name}")
            
            # Extract features (assume same structure as training data)
            feature_names = model_data['feature_names']
            
            # Check if all required features are present
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                print(f"❌ Missing features in CSV: {missing_features}")
                return None
            
            X = df[feature_names].values
            
            # Scale features (using simple standardization)
            # Note: In production, you'd want to save and reuse the scaler from training
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Make predictions
            print("🔮 Making predictions...")
            model = model_data['model']
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            # Create results DataFrame
            results_df = df.copy()
            label_names = model_data['label_names']
            
            # Add prediction columns
            for i, label in enumerate(label_names):
                results_df[f'pred_{label}'] = predictions[:, i]
                if probabilities is not None:
                    results_df[f'prob_{label}'] = probabilities[:, i]
            
            # Save results if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                print(f"💾 Results saved to: {output_path}")
            
            return predictions, results_df
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None, None
    
    def predict_single_sample(self, model_data: Dict, feature_values: List[float]) -> Dict:
        """Make prediction on a single sample."""
        try:
            feature_names = model_data['feature_names']
            label_names = model_data['label_names']
            
            if len(feature_values) != len(feature_names):
                print(f"❌ Expected {len(feature_names)} features, got {len(feature_values)}")
                return None
            
            # Prepare input
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale (simple standardization)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Predict
            model = model_data['model']
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Format results
            results = {
                'input_features': dict(zip(feature_names, feature_values)),
                'predictions': dict(zip(label_names, prediction.astype(int))),
            }
            
            if probabilities is not None:
                results['probabilities'] = dict(zip(label_names, probabilities))
            
            return results
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def evaluate_predictions(self, model_data: Dict, csv_path: Path, 
                           true_labels_cols: List[str]) -> Dict:
        """Evaluate model predictions against true labels."""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Get features and true labels
            feature_names = model_data['feature_names']
            label_names = model_data['label_names']
            
            X = df[feature_names].values
            y_true = df[true_labels_cols].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Predict
            model = model_data['model']
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics
            from sklearn.metrics import hamming_loss, jaccard_score
            
            metrics = {
                'hamming_loss': hamming_loss(y_true, y_pred),
                'jaccard_score_micro': jaccard_score(y_true, y_pred, average='micro'),
                'jaccard_score_macro': jaccard_score(y_true, y_pred, average='macro'),
                'f1_score_micro': f1_score(y_true, y_pred, average='micro'),
                'f1_score_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_score_weighted': f1_score(y_true, y_pred, average='weighted'),
            }
            
            print("\\n📊 Evaluation Results:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating predictions: {e}")
            return None
    
    def run_interactive_inference(self):
        """Interactive inference workflow."""
        print("\\n🔮 LIFT Model Inference System")
        print("="*50)
        
        # List available models
        models = self.list_available_models()
        if not models:
            print("❌ No trained models found!")
            return
        
        print("\\nAvailable Models:")
        print("-" * 40)
        for i, model_info in enumerate(models, 1):
            print(f"{i:2d}. {model_info['dataset']} - {model_info['timestamp']}")
            print(f"    File: {model_info['file']}")
            print(f"    Features: {model_info['feature_count']}, Labels: {model_info['label_count']}")
        
        # Get user choice
        while True:
            try:
                choice = input("\\n🔢 Select model number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    return
                
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    selected_model = models[model_idx]
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        # Load selected model
        model_data = self.load_model(selected_model['path'])
        if not model_data:
            return
        
        # Choose inference type
        print("\\n🎯 Inference Options:")
        print("1. Single sample prediction")
        print("2. Batch prediction from CSV")
        print("3. Evaluate on test data")
        
        while True:
            try:
                inference_choice = int(input("\\nSelect option (1-3): "))
                if 1 <= inference_choice <= 3:
                    break
                else:
                    print("❌ Please enter 1, 2, or 3.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        if inference_choice == 1:
            self.handle_single_prediction(model_data)
        elif inference_choice == 2:
            self.handle_batch_prediction(model_data)
        elif inference_choice == 3:
            self.handle_evaluation(model_data)
    
    def handle_single_prediction(self, model_data: Dict):
        """Handle single sample prediction."""
        feature_names = model_data['feature_names']
        
        print(f"\\n📝 Enter values for {len(feature_names)} features:")
        feature_values = []
        
        for feature_name in feature_names:
            while True:
                try:
                    value = float(input(f"   {feature_name}: "))
                    feature_values.append(value)
                    break
                except ValueError:
                    print("❌ Please enter a numeric value.")
        
        # Make prediction
        result = self.predict_single_sample(model_data, feature_values)
        if result:
            print("\\n🔮 Prediction Results:")
            print("-" * 30)
            for label, prediction in result['predictions'].items():
                status = "✅ YES" if prediction == 1 else "❌ NO"
                print(f"{label}: {status}")
                
                if 'probabilities' in result:
                    prob = result['probabilities'][label]
                    print(f"   Probability: {prob:.4f}")
    
    def handle_batch_prediction(self, model_data: Dict):
        """Handle batch prediction from CSV."""
        csv_path = input("\\n📁 Enter path to CSV file: ").strip()
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return
        
        # Generate output path
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.predictions_dir / f"predictions_{model_data['dataset_name']}_{timestamp}.csv"
        
        # Make predictions
        predictions, results_df = self.predict_from_csv(model_data, csv_path, output_path)
        if predictions is not None:
            print(f"\\n✅ Successfully processed {len(predictions)} samples")
            print(f"💾 Results saved to: {output_path}")
    
    def handle_evaluation(self, model_data: Dict):
        """Handle evaluation on test data."""
        csv_path = input("\\n📁 Enter path to test CSV file: ").strip()
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            return
        
        # Ask for true label columns
        label_names = model_data['label_names']
        print(f"\\n🏷️  Expected label columns: {', '.join(label_names)}")
        use_default = input("Use these column names? (y/n) [y]: ").strip().lower()
        
        if use_default != 'n':
            true_labels_cols = label_names
        else:
            print("Enter true label column names (comma-separated):")
            cols_input = input().strip()
            true_labels_cols = [col.strip() for col in cols_input.split(',')]
        
        # Evaluate
        metrics = self.evaluate_predictions(model_data, csv_path, true_labels_cols)
        if metrics:
            print("\\n✅ Evaluation completed successfully!")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="LIFT Model Inference System")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--model", "-m", type=str,
                       help="Path to trained model file")
    parser.add_argument("--data", "-d", type=str,
                       help="Path to CSV file for prediction")
    parser.add_argument("--output", "-o", type=str,
                       help="Output path for predictions")
    parser.add_argument("--evaluate", "-e", action="store_true",
                       help="Evaluate model on test data")
    
    args = parser.parse_args()
    
    inference = LIFTInference()
    
    if args.interactive or (not args.model and not args.data):
        inference.run_interactive_inference()
    else:
        # Non-interactive mode
        if not args.model:
            print("❌ Model path required for non-interactive mode")
            return
        
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return
        
        # Load model
        model_data = inference.load_model(model_path)
        if not model_data:
            return
        
        if args.data:
            data_path = Path(args.data)
            if not data_path.exists():
                print(f"❌ Data file not found: {data_path}")
                return
            
            output_path = Path(args.output) if args.output else None
            
            if args.evaluate:
                # Evaluation mode
                label_names = model_data['label_names']
                metrics = inference.evaluate_predictions(model_data, data_path, label_names)
                if metrics:
                    print("✅ Evaluation completed!")
            else:
                # Prediction mode
                predictions, results_df = inference.predict_from_csv(model_data, data_path, output_path)
                if predictions is not None:
                    print(f"✅ Processed {len(predictions)} samples successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Script for Multilabel Training System

This script performs basic tests to ensure the system is working correctly.

Author: GitHub Copilot
"""

import sys
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd


def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import numpy
        print("   ✅ NumPy")
    except ImportError:
        print("   ❌ NumPy")
        return False
    
    try:
        import pandas
        print("   ✅ Pandas")
    except ImportError:
        print("   ❌ Pandas")
        return False
    
    try:
        import sklearn
        print("   ✅ Scikit-learn")
    except ImportError:
        print("   ❌ Scikit-learn")
        return False
    
    try:
        import skopt
        print("   ✅ Scikit-optimize")
    except ImportError:
        print("   ❌ Scikit-optimize")
        return False
    
    # Test LIFT package
    try:
        sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')
        from lift_ml import LIFTClassifier
        print("   ✅ LIFT package")
    except ImportError as e:
        print(f"   ❌ LIFT package: {e}")
        return False
    
    return True


def test_lift_functionality():
    """Test basic LIFT functionality with synthetic data."""
    print("\\n🧪 Testing LIFT functionality...")
    
    try:
        from sklearn.datasets import make_multilabel_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from lift_ml import LIFTClassifier
        
        # Generate synthetic data
        X, y = make_multilabel_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            n_labels=2,
            random_state=42,
            return_indicator=True
        )
        
        print(f"   Generated synthetic data: {X.shape} features, {y.shape} labels")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LIFT classifier
        clf = LIFTClassifier(k=2, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test_scaled)
        
        print(f"   ✅ LIFT classifier trained and predictions made")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Prediction shape: {y_pred.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LIFT functionality test failed: {e}")
        return False


def test_dataset_structure():
    """Test that datasets are available and properly structured."""
    print("\\n📁 Testing dataset availability...")
    
    datasets = ['birds', 'bookmarks', 'Cal500', 'corel5k', 'delicious',
               'Emotions', 'enron', 'genbase', 'mediamill', 'yeast']
    
    available_count = 0
    for dataset in datasets:
        zip_path = Path(f"{dataset}.zip")
        if zip_path.exists():
            print(f"   ✅ {dataset}.zip")
            available_count += 1
        else:
            print(f"   ❌ {dataset}.zip (missing)")
    
    print(f"   📊 Available datasets: {available_count}/{len(datasets)}")
    
    if available_count > 0:
        print("   ✅ At least one dataset is available")
        return True
    else:
        print("   ❌ No datasets found")
        return False


def test_script_existence():
    """Test that all main scripts exist."""
    print("\\n📜 Testing script availability...")
    
    scripts = [
        'multilabel_trainer.py',
        'lift_inference.py', 
        'dataset_explorer.py',
        'batch_runner.py',
        'run_lift_experiment.py',
        'quickstart.py',
        'setup.py'
    ]
    
    all_exist = True
    for script in scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script}")
            all_exist = False
    
    return all_exist


def test_directories():
    """Test that necessary directories can be created."""
    print("\\n📁 Testing directory creation...")
    
    test_dirs = [
        'extracted_datasets',
        'trained_models',
        'reports', 
        'dataset_reports',
        'predictions'
    ]
    
    try:
        for dir_name in test_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/")
        
        print("   ✅ All directories created successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Directory creation failed: {e}")
        return False


def create_test_data():
    """Create a small test dataset for testing."""
    print("\\n🔬 Creating test dataset...")
    
    try:
        # Create synthetic multilabel data
        np.random.seed(42)
        n_samples = 50
        n_features = 5
        n_labels = 3
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.binomial(1, 0.3, (n_samples, n_labels))
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        label_cols = [f'label_{i}' for i in range(n_labels)]
        
        df = pd.DataFrame(X, columns=feature_cols)
        for i, col in enumerate(label_cols):
            df[col] = y[:, i]
        
        # Save test data
        test_dir = Path('test_data')
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / 'synthetic_test.csv'
        df.to_csv(test_file, index=False)
        
        print(f"   ✅ Test dataset created: {test_file}")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {n_features}, Labels: {n_labels}")
        
        return test_file
        
    except Exception as e:
        print(f"   ❌ Test data creation failed: {e}")
        return None


def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 MULTILABEL TRAINING SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("LIFT Functionality Test", test_lift_functionality),
        ("Dataset Availability Test", test_dataset_structure),
        ("Script Existence Test", test_script_existence),
        ("Directory Creation Test", test_directories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\n🔬 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Create test data
    test_file = create_test_data()
    if test_file:
        results.append(("Test Data Creation", True))
    else:
        results.append(("Test Data Creation", False))
    
    # Summary
    print("\\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
    
    print(f"\\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\\n🚀 Try running: python quickstart.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Try running: python setup.py")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

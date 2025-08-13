#!/usr/bin/env python3
"""
Setup and Installation Script for Multilabel Dataset Training System

This script sets up the environment and installs necessary dependencies
for the LIFT multilabel training system.

Author: GitHub Copilot
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_requirements():
    """Install Python package requirements."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def setup_lift_package():
    """Set up the LIFT package."""
    lift_dir = Path("LIFT-MultiLabel-Learning-with-Label-Specific-Features")
    
    if not lift_dir.exists():
        print("❌ LIFT package directory not found!")
        print("   Make sure the LIFT submodule is properly initialized.")
        return False
    
    # Install LIFT package in development mode
    original_dir = os.getcwd()
    try:
        os.chdir(lift_dir)
        success = run_command(
            f"{sys.executable} -m pip install -e .",
            "Installing LIFT package"
        )
        return success
    finally:
        os.chdir(original_dir)


def create_directories():
    """Create necessary directories."""
    directories = [
        "extracted_datasets",
        "trained_models", 
        "reports",
        "dataset_reports",
        "predictions"
    ]
    
    print("📁 Creating necessary directories...")
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    return True


def verify_installation():
    """Verify that the installation was successful."""
    print("\\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"), 
        ("sklearn", "Scikit-learn"),
        ("skopt", "Scikit-optimize")
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            all_good = False
    
    # Test LIFT package
    try:
        sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')
        from lift_ml import LIFTClassifier
        print(f"   ✅ LIFT package")
    except ImportError as e:
        print(f"   ❌ LIFT package: {e}")
        all_good = False
    
    return all_good


def main():
    """Main setup function."""
    print("🚀 Setting up Multilabel Dataset Training System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Setup LIFT package
    if not setup_lift_package():
        print("❌ Failed to setup LIFT package")
        print("\\n💡 Troubleshooting:")
        print("   1. Make sure the LIFT submodule is initialized:")
        print("      git submodule update --init --recursive")
        print("   2. Check if the LIFT directory exists and contains source code")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\\n❌ Installation verification failed")
        sys.exit(1)
    
    print("\\n🎉 Setup completed successfully!")
    print("\\n📚 Available Scripts:")
    print("   🔧 multilabel_trainer.py    - Interactive training system")
    print("   🔮 lift_inference.py        - Model inference and evaluation")
    print("   🔍 dataset_explorer.py      - Dataset analysis and comparison")
    
    print("\\n🚀 Quick Start:")
    print("   python multilabel_trainer.py --interactive")
    print("   python dataset_explorer.py --interactive")
    print("   python lift_inference.py --interactive")
    
    print("\\n📖 For more information, see the documentation in each script.")


if __name__ == "__main__":
    main()

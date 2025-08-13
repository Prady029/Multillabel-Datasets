#!/usr/bin/env python3
"""
Quick Start Script for Multilabel Training System

This script provides a simple menu interface to get started quickly
with the multilabel training system.

Author: GitHub Copilot
"""

import os
import sys
import subprocess
from pathlib import Path


def check_setup():
    """Check if the system is properly set up."""
    print("🔍 Checking system setup...")
    
    # Check if LIFT directory exists
    lift_dir = Path("LIFT-MultiLabel-Learning-with-Label-Specific-Features")
    if not lift_dir.exists():
        print("❌ LIFT package directory not found!")
        print("Please run: git submodule update --init --recursive")
        return False
    
    # Check if main scripts exist
    scripts = ["multilabel_trainer.py", "lift_inference.py", "dataset_explorer.py"]
    for script in scripts:
        if not Path(script).exists():
            print(f"❌ Script {script} not found!")
            return False
    
    # Check if any datasets are available
    datasets = ['birds', 'bookmarks', 'Cal500', 'corel5k', 'delicious',
               'Emotions', 'enron', 'genbase', 'mediamill', 'yeast']
    
    available_datasets = []
    for dataset in datasets:
        if Path(f"{dataset}.zip").exists():
            available_datasets.append(dataset)
    
    if not available_datasets:
        print("❌ No dataset files found!")
        print("Please ensure dataset .zip files are in the current directory.")
        return False
    
    print(f"✅ Found {len(available_datasets)} dataset(s)")
    print("✅ System appears to be set up correctly!")
    return True


def run_setup():
    """Run the setup script."""
    print("🔧 Running setup...")
    try:
        result = subprocess.run([sys.executable, "setup.py"], check=True)
        print("✅ Setup completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Setup failed!")
        return False


def show_menu():
    """Show the main menu."""
    print("\n" + "="*60)
    print("🚀 MULTILABEL TRAINING SYSTEM - QUICK START")
    print("="*60)
    print()
    print("Choose an option:")
    print()
    print("1. 🔧 Setup System (install dependencies)")
    print("2. 🎯 Interactive Training (train models)")
    print("3. 🔮 Model Inference (use trained models)")
    print("4. 🔍 Dataset Explorer (analyze datasets)")
    print("5. 📊 Batch Runner (train multiple datasets)")
    print("6. 🏃 Quick Example (run simple experiment)")
    print("7. 📖 Help & Documentation")
    print("8. 🚪 Exit")
    print()


def run_script(script_name, args=None):
    """Run a script with optional arguments."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Interrupted execution of {script_name}")


def show_help():
    """Show help information."""
    print("\n" + "="*60)
    print("📖 HELP & DOCUMENTATION")
    print("="*60)
    print()
    print("📚 Available Scripts:")
    print("   multilabel_trainer.py  - Main training interface")
    print("   lift_inference.py      - Model inference and evaluation")
    print("   dataset_explorer.py    - Dataset analysis and comparison")
    print("   batch_runner.py        - Batch training experiments")
    print("   run_lift_experiment.py - Simple example script")
    print()
    print("📄 Documentation Files:")
    print("   USAGE_GUIDE.md         - Comprehensive usage guide")
    print("   README.md              - Basic project information")
    print("   requirements.txt       - Python dependencies")
    print()
    print("🔧 Command Line Examples:")
    print("   python multilabel_trainer.py --interactive")
    print("   python dataset_explorer.py --all")
    print("   python lift_inference.py --interactive")
    print("   python batch_runner.py --comparison")
    print()
    print("🌐 For detailed documentation, see USAGE_GUIDE.md")
    print()


def main():
    """Main quick start interface."""
    print("👋 Welcome to the LIFT Multilabel Training System!")
    
    # Check if setup is needed
    if not check_setup():
        print("\n🔧 System setup required.")
        setup_choice = input("Run setup now? (y/n) [y]: ").strip().lower()
        if setup_choice != 'n':
            if not run_setup():
                print("❌ Setup failed. Please check the errors above.")
                return
        else:
            print("⚠️  Please run setup before using the system.")
            return
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                run_setup()
            
            elif choice == '2':
                print("\n🎯 Starting Interactive Training...")
                run_script("multilabel_trainer.py", ["--interactive"])
            
            elif choice == '3':
                print("\n🔮 Starting Model Inference...")
                run_script("lift_inference.py", ["--interactive"])
            
            elif choice == '4':
                print("\n🔍 Starting Dataset Explorer...")
                run_script("dataset_explorer.py", ["--interactive"])
            
            elif choice == '5':
                print("\n📊 Starting Batch Runner...")
                print("Choose batch mode:")
                print("  1. Run all datasets with optimization")
                print("  2. Run all datasets without optimization")
                print("  3. Run comparison experiment")
                
                batch_choice = input("Enter choice (1-3): ").strip()
                if batch_choice == '1':
                    run_script("batch_runner.py", [])
                elif batch_choice == '2':
                    run_script("batch_runner.py", ["--no-optimization"])
                elif batch_choice == '3':
                    run_script("batch_runner.py", ["--comparison"])
                else:
                    print("❌ Invalid choice.")
            
            elif choice == '6':
                print("\n🏃 Running Quick Example...")
                dataset = input("Enter dataset name (or press Enter for 'yeast'): ").strip()
                if not dataset:
                    dataset = "yeast"
                run_script("run_lift_experiment.py", ["--dataset", dataset])
            
            elif choice == '7':
                show_help()
            
            elif choice == '8':
                print("\n👋 Thank you for using the LIFT Multilabel Training System!")
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()

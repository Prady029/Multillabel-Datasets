#!/usr/bin/env python3
"""
Batch Runner for Multilabel Experiments

This script allows running training experiments on multiple datasets
in batch mode with different configurations.

Author: GitHub Copilot
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add LIFT package to path
sys.path.append('./LIFT-MultiLabel-Learning-with-Label-Specific-Features/src')


class BatchRunner:
    """Class for running batch experiments on multiple datasets."""
    
    def __init__(self):
        self.workspace_dir = Path('.')
        self.results_dir = self.workspace_dir / 'batch_results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.available_datasets = [
            'birds', 'bookmarks', 'Cal500', 'corel5k', 'delicious',
            'Emotions', 'enron', 'genbase', 'mediamill', 'yeast'
        ]
    
    def check_dataset_availability(self) -> list:
        """Check which datasets are available."""
        available = []
        for dataset in self.available_datasets:
            zip_path = self.workspace_dir / f"{dataset}.zip"
            if zip_path.exists():
                available.append(dataset)
        return available
    
    def run_single_experiment(self, dataset: str, optimize: bool = True) -> dict:
        """Run a single training experiment."""
        print(f"\\n🔧 Running experiment on {dataset}...")
        
        # Prepare command
        cmd = [
            sys.executable, 'multilabel_trainer.py',
            '--dataset', dataset
        ]
        
        if optimize:
            cmd.append('--optimize')
        
        try:
            # Run training
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully trained on {dataset}")
                return {
                    'dataset': dataset,
                    'status': 'success',
                    'optimize': optimize,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"❌ Failed to train on {dataset}")
                print(f"   Error: {result.stderr}")
                return {
                    'dataset': dataset,
                    'status': 'failed',
                    'optimize': optimize,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout training on {dataset}")
            return {
                'dataset': dataset,
                'status': 'timeout',
                'optimize': optimize,
                'stdout': '',
                'stderr': 'Training timeout after 1 hour'
            }
        
        except Exception as e:
            print(f"❌ Error training on {dataset}: {e}")
            return {
                'dataset': dataset,
                'status': 'error',
                'optimize': optimize,
                'stdout': '',
                'stderr': str(e)
            }
    
    def run_batch_experiments(self, datasets: list = None, optimize: bool = True) -> dict:
        """Run experiments on multiple datasets."""
        if datasets is None:
            datasets = self.check_dataset_availability()
        
        print(f"🚀 Starting batch experiments on {len(datasets)} datasets")
        print(f"📊 Datasets: {', '.join(datasets)}")
        print(f"⚡ Optimization: {'Enabled' if optimize else 'Disabled'}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_datasets': len(datasets),
            'optimization_enabled': optimize,
            'experiments': []
        }
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\\n[{i}/{len(datasets)}] Processing {dataset}...")
            
            experiment_result = self.run_single_experiment(dataset, optimize)
            results['experiments'].append(experiment_result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"batch_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n📄 Results saved to: {results_file}")
        
        # Summary
        successful = len([r for r in results['experiments'] if r['status'] == 'success'])
        failed = len([r for r in results['experiments'] if r['status'] != 'success'])
        
        print(f"\\n📊 Batch Experiment Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {successful/len(datasets)*100:.1f}%")
        
        return results
    
    def run_comparison_experiment(self, datasets: list = None) -> dict:
        """Run experiments with and without optimization for comparison."""
        if datasets is None:
            datasets = self.check_dataset_availability()
        
        print(f"🔬 Running comparison experiments (with/without optimization)")
        print(f"📊 Datasets: {', '.join(datasets)}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'comparison',
            'datasets': datasets,
            'with_optimization': [],
            'without_optimization': []
        }
        
        # Run without optimization first (faster)
        print("\\n🔧 Phase 1: Training without optimization")
        for i, dataset in enumerate(datasets, 1):
            print(f"[{i}/{len(datasets)}] {dataset} (no optimization)...")
            result = self.run_single_experiment(dataset, optimize=False)
            results['without_optimization'].append(result)
        
        # Run with optimization
        print("\\n⚡ Phase 2: Training with optimization")
        for i, dataset in enumerate(datasets, 1):
            print(f"[{i}/{len(datasets)}] {dataset} (with optimization)...")
            result = self.run_single_experiment(dataset, optimize=True)
            results['with_optimization'].append(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comparison_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n📄 Comparison results saved to: {results_file}")
        
        return results
    
    def analyze_batch_results(self, results_file: Path) -> None:
        """Analyze results from a batch experiment."""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\\n📊 Analyzing results from {results_file.name}")
            print("="*60)
            
            if results.get('experiment_type') == 'comparison':
                # Comparison analysis
                print("🔬 Comparison Experiment Analysis")
                
                no_opt = results['without_optimization']
                with_opt = results['with_optimization']
                
                no_opt_success = len([r for r in no_opt if r['status'] == 'success'])
                with_opt_success = len([r for r in with_opt if r['status'] == 'success'])
                
                print(f"\\nWithout Optimization:")
                print(f"   ✅ Successful: {no_opt_success}/{len(no_opt)}")
                print(f"   📈 Success Rate: {no_opt_success/len(no_opt)*100:.1f}%")
                
                print(f"\\nWith Optimization:")
                print(f"   ✅ Successful: {with_opt_success}/{len(with_opt)}")
                print(f"   📈 Success Rate: {with_opt_success/len(with_opt)*100:.1f}%")
                
            else:
                # Regular batch analysis
                experiments = results.get('experiments', [])
                total = len(experiments)
                successful = len([r for r in experiments if r['status'] == 'success'])
                failed = len([r for r in experiments if r['status'] == 'failed'])
                timeout = len([r for r in experiments if r['status'] == 'timeout'])
                error = len([r for r in experiments if r['status'] == 'error'])
                
                print(f"📈 Batch Experiment Results:")
                print(f"   Total Datasets: {total}")
                print(f"   ✅ Successful: {successful}")
                print(f"   ❌ Failed: {failed}")
                print(f"   ⏰ Timeout: {timeout}")
                print(f"   🔥 Error: {error}")
                print(f"   📊 Success Rate: {successful/total*100:.1f}%")
                
                if failed > 0 or timeout > 0 or error > 0:
                    print(f"\\n❌ Failed Experiments:")
                    for exp in experiments:
                        if exp['status'] != 'success':
                            print(f"   {exp['dataset']}: {exp['status']}")
                            if exp['stderr']:
                                print(f"      Error: {exp['stderr'][:100]}...")
        
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Batch Runner for Multilabel Experiments")
    parser.add_argument("--datasets", "-d", nargs='+',
                       help="Specific datasets to run experiments on")
    parser.add_argument("--no-optimization", action="store_true",
                       help="Disable Bayesian optimization")
    parser.add_argument("--comparison", "-c", action="store_true",
                       help="Run comparison experiment (with/without optimization)")
    parser.add_argument("--analyze", "-a", type=str,
                       help="Analyze results from a previous batch run")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    runner = BatchRunner()
    
    if args.list:
        # List available datasets
        available = runner.check_dataset_availability()
        print("Available Datasets:")
        for dataset in available:
            print(f"  ✅ {dataset}")
        
        missing = set(runner.available_datasets) - set(available)
        if missing:
            print("\\nMissing Datasets:")
            for dataset in missing:
                print(f"  ❌ {dataset}")
    
    elif args.analyze:
        # Analyze previous results
        results_file = Path(args.analyze)
        if results_file.exists():
            runner.analyze_batch_results(results_file)
        else:
            print(f"❌ Results file not found: {results_file}")
    
    elif args.comparison:
        # Run comparison experiment
        datasets = args.datasets if args.datasets else None
        runner.run_comparison_experiment(datasets)
    
    else:
        # Run batch experiments
        datasets = args.datasets if args.datasets else None
        optimize = not args.no_optimization
        
        runner.run_batch_experiments(datasets, optimize)


if __name__ == "__main__":
    main()

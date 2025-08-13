#!/usr/bin/env python3
"""
Dataset Explorer and Comparison Tool

This script provides functionality to:
1. Explore multilabel datasets
2. Generate dataset statistics and visualizations
3. Compare multiple datasets
4. Export dataset summaries

Author: GitHub Copilot
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datetime import datetime


class DatasetExplorer:
    """Class for exploring and analyzing multilabel datasets."""
    
    def __init__(self):
        self.workspace_dir = Path('.')
        self.data_dir = self.workspace_dir / 'extracted_datasets'
        self.reports_dir = self.workspace_dir / 'dataset_reports'
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.datasets_info = {
            'birds': 'Bird species classification from audio features',
            'bookmarks': 'Web bookmark categorization',
            'Cal500': 'Music emotion classification (CAL500)',
            'corel5k': 'Image annotation with Corel 5K dataset',
            'delicious': 'Social bookmarking tag prediction',
            'Emotions': 'Music emotion classification',
            'enron': 'Email classification (Enron dataset)',
            'genbase': 'Gene functional classification',
            'mediamill': 'Video semantic annotation',
            'yeast': 'Yeast protein functional classification'
        }
    
    def extract_dataset_if_needed(self, dataset_name: str) -> bool:
        """Extract dataset if not already extracted."""
        zip_path = self.workspace_dir / f"{dataset_name}.zip"
        extract_path = self.data_dir / dataset_name
        
        if not zip_path.exists():
            print(f"❌ Dataset {dataset_name} not found!")
            return False
        
        if extract_path.exists():
            return True
        
        try:
            print(f"📦 Extracting {dataset_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            return True
        except Exception as e:
            print(f"❌ Error extracting {dataset_name}: {e}")
            return False
    
    def analyze_dataset(self, dataset_name: str) -> Optional[Dict]:
        """Comprehensive analysis of a multilabel dataset."""
        if not self.extract_dataset_if_needed(dataset_name):
            return None
        
        extract_path = self.data_dir / dataset_name
        
        # Find CSV files
        csv_files = list(extract_path.rglob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {dataset_name}")
            return None
        
        print(f"\\n🔍 Analyzing dataset: {dataset_name}")
        print(f"📄 Found {len(csv_files)} CSV files")
        
        analysis = {
            'dataset_name': dataset_name,
            'description': self.datasets_info.get(dataset_name, 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'files': [],
            'summary': {}
        }
        
        total_samples = 0
        all_features = set()
        all_labels = set()
        
        for csv_file in csv_files:
            try:
                print(f"   📊 Processing {csv_file.name}...")
                df = pd.read_csv(csv_file)
                
                # Auto-detect features and labels
                binary_cols = []
                for col in df.columns:
                    unique_vals = set(df[col].dropna().unique())
                    if unique_vals.issubset({0, 1, 0.0, 1.0}):
                        binary_cols.append(col)
                
                # Heuristic for label detection
                if len(binary_cols) > len(df.columns) // 2:
                    label_cols = binary_cols
                    feature_cols = [col for col in df.columns if col not in label_cols]
                else:
                    n_features = len(df.columns) - len(binary_cols) if binary_cols else len(df.columns) - 5
                    n_features = max(1, n_features)
                    feature_cols = df.columns[:n_features].tolist()
                    label_cols = df.columns[n_features:].tolist()
                
                # File analysis
                file_analysis = {
                    'filename': csv_file.name,
                    'samples': len(df),
                    'total_columns': len(df.columns),
                    'feature_columns': len(feature_cols),
                    'label_columns': len(label_cols),
                    'features': feature_cols,
                    'labels': label_cols,
                    'missing_values': df.isnull().sum().sum(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                
                # Label statistics
                if label_cols:
                    y = df[label_cols].values
                    file_analysis['label_stats'] = {
                        'label_cardinality': float(np.mean(np.sum(y, axis=1))),  # avg labels per sample
                        'label_density': float(np.mean(y)),  # proportion of positive labels
                        'label_frequencies': {label: float(df[label].mean()) for label in label_cols},
                        'distinct_labelsets': int(len(set(tuple(row) for row in y)))
                    }
                
                # Feature statistics
                if feature_cols:
                    X = df[feature_cols]
                    file_analysis['feature_stats'] = {
                        'numeric_features': len([col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]),
                        'categorical_features': len(feature_cols) - len([col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]),
                        'feature_ranges': {},
                        'feature_correlations': {}
                    }
                    
                    # Feature ranges for numeric columns
                    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
                    for col in numeric_cols[:10]:  # Limit to first 10 for brevity
                        file_analysis['feature_stats']['feature_ranges'][col] = {
                            'min': float(X[col].min()),
                            'max': float(X[col].max()),
                            'mean': float(X[col].mean()),
                            'std': float(X[col].std())
                        }
                
                analysis['files'].append(file_analysis)
                total_samples += len(df)
                all_features.update(feature_cols)
                all_labels.update(label_cols)
                
            except Exception as e:
                print(f"   ⚠️  Error processing {csv_file.name}: {e}")
        
        # Overall summary
        analysis['summary'] = {
            'total_samples': total_samples,
            'unique_features': len(all_features),
            'unique_labels': len(all_labels),
            'files_count': len(analysis['files']),
            'avg_samples_per_file': total_samples / len(analysis['files']) if analysis['files'] else 0
        }
        
        # Calculate dataset-level label statistics
        if analysis['files']:
            all_cardinalities = []
            all_densities = []
            all_label_freqs = {}
            
            for file_info in analysis['files']:
                if 'label_stats' in file_info:
                    all_cardinalities.append(file_info['label_stats']['label_cardinality'])
                    all_densities.append(file_info['label_stats']['label_density'])
                    
                    for label, freq in file_info['label_stats']['label_frequencies'].items():
                        if label not in all_label_freqs:
                            all_label_freqs[label] = []
                        all_label_freqs[label].append(freq)
            
            if all_cardinalities:
                analysis['summary']['label_cardinality_avg'] = float(np.mean(all_cardinalities))
                analysis['summary']['label_density_avg'] = float(np.mean(all_densities))
                analysis['summary']['most_frequent_labels'] = sorted(
                    [(label, np.mean(freqs)) for label, freqs in all_label_freqs.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
        
        return analysis
    
    def generate_comparison_report(self, dataset_analyses: List[Dict]) -> Dict:
        """Generate a comparison report for multiple datasets."""
        if not dataset_analyses:
            return {}
        
        print("\\n📊 Generating comparison report...")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'datasets_compared': len(dataset_analyses),
            'comparison_metrics': {},
            'rankings': {}
        }
        
        # Extract metrics for comparison
        metrics = []
        for analysis in dataset_analyses:
            if 'summary' in analysis:
                summary = analysis['summary']
                metrics.append({
                    'name': analysis['dataset_name'],
                    'samples': summary.get('total_samples', 0),
                    'features': summary.get('unique_features', 0),
                    'labels': summary.get('unique_labels', 0),
                    'cardinality': summary.get('label_cardinality_avg', 0),
                    'density': summary.get('label_density_avg', 0),
                    'files': summary.get('files_count', 0)
                })
        
        # Calculate statistics
        if metrics:
            comparison['comparison_metrics'] = {
                'samples': {
                    'min': min(m['samples'] for m in metrics),
                    'max': max(m['samples'] for m in metrics),
                    'avg': np.mean([m['samples'] for m in metrics]),
                    'std': np.std([m['samples'] for m in metrics])
                },
                'features': {
                    'min': min(m['features'] for m in metrics),
                    'max': max(m['features'] for m in metrics),
                    'avg': np.mean([m['features'] for m in metrics]),
                    'std': np.std([m['features'] for m in metrics])
                },
                'labels': {
                    'min': min(m['labels'] for m in metrics),
                    'max': max(m['labels'] for m in metrics),
                    'avg': np.mean([m['labels'] for m in metrics]),
                    'std': np.std([m['labels'] for m in metrics])
                }
            }
            
            # Rankings
            comparison['rankings'] = {
                'by_samples': sorted(metrics, key=lambda x: x['samples'], reverse=True),
                'by_features': sorted(metrics, key=lambda x: x['features'], reverse=True),
                'by_labels': sorted(metrics, key=lambda x: x['labels'], reverse=True),
                'by_complexity': sorted(metrics, key=lambda x: x['samples'] * x['features'] * x['labels'], reverse=True)
            }
        
        return comparison
    
    def export_analysis_report(self, analysis: Dict) -> None:
        """Export analysis to JSON and HTML formats."""
        dataset_name = analysis['dataset_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = self.reports_dir / f"{dataset_name}_analysis_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"📄 JSON report saved: {json_path}")
        
        # HTML report
        html_path = self.reports_dir / f"{dataset_name}_analysis_{timestamp}.html"
        self.generate_html_report(analysis, html_path)
        print(f"🌐 HTML report saved: {html_path}")
    
    def generate_html_report(self, analysis: Dict, output_path: Path) -> None:
        """Generate HTML report for dataset analysis."""
        dataset_name = analysis['dataset_name']
        summary = analysis.get('summary', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Analysis - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .files {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dataset Analysis: {dataset_name}</h1>
                <p><strong>Description:</strong> {analysis.get('description', 'N/A')}</p>
                <p><strong>Analysis Date:</strong> {analysis.get('timestamp', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Summary</h2>
                <div class="metric"><strong>Total Samples:</strong> {summary.get('total_samples', 'N/A'):,}</div>
                <div class="metric"><strong>Unique Features:</strong> {summary.get('unique_features', 'N/A')}</div>
                <div class="metric"><strong>Unique Labels:</strong> {summary.get('unique_labels', 'N/A')}</div>
                <div class="metric"><strong>Number of Files:</strong> {summary.get('files_count', 'N/A')}</div>
                <div class="metric"><strong>Average Label Cardinality:</strong> {summary.get('label_cardinality_avg', 0):.3f}</div>
                <div class="metric"><strong>Average Label Density:</strong> {summary.get('label_density_avg', 0):.3f}</div>
            </div>
        """
        
        # Most frequent labels
        if 'most_frequent_labels' in summary:
            html_content += """
            <div class="section">
                <h2>Most Frequent Labels</h2>
                <table>
                    <tr><th>Label</th><th>Frequency</th></tr>
            """
            for label, freq in summary['most_frequent_labels']:
                html_content += f"<tr><td>{label}</td><td>{freq:.3f}</td></tr>"
            html_content += "</table></div>"
        
        # File details
        html_content += "<div class='section'><h2>File Details</h2>"
        for file_info in analysis.get('files', []):
            html_content += f"""
            <div class="files">
                <h3>{file_info['filename']}</h3>
                <p><strong>Samples:</strong> {file_info['samples']:,}</p>
                <p><strong>Features:</strong> {file_info['feature_columns']}</p>
                <p><strong>Labels:</strong> {file_info['label_columns']}</p>
                <p><strong>Missing Values:</strong> {file_info['missing_values']}</p>
                <p><strong>Memory Usage:</strong> {file_info['memory_usage_mb']:.2f} MB</p>
            """
            
            if 'label_stats' in file_info:
                stats = file_info['label_stats']
                html_content += f"""
                <p><strong>Label Cardinality:</strong> {stats['label_cardinality']:.3f}</p>
                <p><strong>Label Density:</strong> {stats['label_density']:.3f}</p>
                <p><strong>Distinct Labelsets:</strong> {stats['distinct_labelsets']}</p>
                """
            
            html_content += "</div>"
        
        html_content += "</div></body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def run_interactive_exploration(self):
        """Interactive dataset exploration workflow."""
        print("\\n🔍 Dataset Explorer")
        print("="*50)
        
        available_datasets = [name for name in self.datasets_info.keys() 
                            if (self.workspace_dir / f"{name}.zip").exists()]
        
        if not available_datasets:
            print("❌ No datasets found!")
            return
        
        print("\\nAvailable Datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"{i:2d}. {dataset} - {self.datasets_info[dataset]}")
        
        print("\\n🎯 Exploration Options:")
        print("1. Analyze single dataset")
        print("2. Compare multiple datasets")
        print("3. Analyze all datasets")
        
        while True:
            try:
                choice = int(input("\\nSelect option (1-3): "))
                if 1 <= choice <= 3:
                    break
                else:
                    print("❌ Please enter 1, 2, or 3.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        if choice == 1:
            # Single dataset analysis
            while True:
                try:
                    dataset_choice = int(input(f"\\nSelect dataset (1-{len(available_datasets)}): ")) - 1
                    if 0 <= dataset_choice < len(available_datasets):
                        dataset_name = available_datasets[dataset_choice]
                        break
                    else:
                        print("❌ Invalid choice.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            analysis = self.analyze_dataset(dataset_name)
            if analysis:
                self.export_analysis_report(analysis)
                print(f"\\n✅ Analysis completed for {dataset_name}!")
        
        elif choice == 2:
            # Multiple dataset comparison
            print("\\nSelect datasets for comparison (comma-separated numbers):")
            selections = input("Dataset numbers: ").strip().split(',')
            
            selected_datasets = []
            for sel in selections:
                try:
                    idx = int(sel.strip()) - 1
                    if 0 <= idx < len(available_datasets):
                        selected_datasets.append(available_datasets[idx])
                except ValueError:
                    continue
            
            if len(selected_datasets) < 2:
                print("❌ Please select at least 2 datasets for comparison.")
                return
            
            analyses = []
            for dataset in selected_datasets:
                analysis = self.analyze_dataset(dataset)
                if analysis:
                    analyses.append(analysis)
            
            if analyses:
                comparison = self.generate_comparison_report(analyses)
                
                # Save comparison report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_path = self.reports_dir / f"comparison_report_{timestamp}.json"
                with open(comp_path, 'w') as f:
                    json.dump({'analyses': analyses, 'comparison': comparison}, f, indent=2)
                
                print(f"\\n✅ Comparison completed for {len(selected_datasets)} datasets!")
                print(f"📄 Report saved: {comp_path}")
        
        elif choice == 3:
            # Analyze all datasets
            print(f"\\n🔄 Analyzing all {len(available_datasets)} datasets...")
            
            analyses = []
            for dataset in available_datasets:
                analysis = self.analyze_dataset(dataset)
                if analysis:
                    analyses.append(analysis)
                    self.export_analysis_report(analysis)
            
            if analyses:
                comparison = self.generate_comparison_report(analyses)
                
                # Save comprehensive report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_path = self.reports_dir / f"all_datasets_report_{timestamp}.json"
                with open(comp_path, 'w') as f:
                    json.dump({'analyses': analyses, 'comparison': comparison}, f, indent=2)
                
                print(f"\\n✅ Analysis completed for all {len(analyses)} datasets!")
                print(f"📄 Comprehensive report saved: {comp_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Dataset Explorer and Analysis Tool")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--dataset", "-d", type=str,
                       help="Dataset name to analyze")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Analyze all available datasets")
    parser.add_argument("--compare", "-c", nargs='+',
                       help="Compare multiple datasets")
    
    args = parser.parse_args()
    
    explorer = DatasetExplorer()
    
    if args.interactive or (not args.dataset and not args.all and not args.compare):
        explorer.run_interactive_exploration()
    else:
        if args.all:
            # Analyze all datasets
            available_datasets = [name for name in explorer.datasets_info.keys() 
                                if (explorer.workspace_dir / f"{name}.zip").exists()]
            
            analyses = []
            for dataset in available_datasets:
                analysis = explorer.analyze_dataset(dataset)
                if analysis:
                    analyses.append(analysis)
                    explorer.export_analysis_report(analysis)
            
            if analyses:
                comparison = explorer.generate_comparison_report(analyses)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_path = explorer.reports_dir / f"all_datasets_report_{timestamp}.json"
                with open(comp_path, 'w') as f:
                    json.dump({'analyses': analyses, 'comparison': comparison}, f, indent=2)
                print(f"✅ Analysis completed for all datasets!")
        
        elif args.dataset:
            # Single dataset analysis
            analysis = explorer.analyze_dataset(args.dataset)
            if analysis:
                explorer.export_analysis_report(analysis)
                print(f"✅ Analysis completed for {args.dataset}!")
        
        elif args.compare:
            # Compare specific datasets
            analyses = []
            for dataset in args.compare:
                analysis = explorer.analyze_dataset(dataset)
                if analysis:
                    analyses.append(analysis)
            
            if len(analyses) >= 2:
                comparison = explorer.generate_comparison_report(analyses)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_path = explorer.reports_dir / f"comparison_report_{timestamp}.json"
                with open(comp_path, 'w') as f:
                    json.dump({'analyses': analyses, 'comparison': comparison}, f, indent=2)
                print(f"✅ Comparison completed for {len(analyses)} datasets!")
            else:
                print("❌ Need at least 2 valid datasets for comparison.")


if __name__ == "__main__":
    main()

# Multilabel Datasets with LIFT Training System

A comprehensive system for training, evaluating, and deploying multilabel classifiers using the LIFT (Learning with Label-Specific Features) algorithm on various multilabel datasets.

## 🚀 Quick Start

Get started quickly with the interactive interface:

```bash
python quickstart.py
```

This will guide you through setup and provide access to all features.

## 📦 Available Datasets

The repository includes 10 popular multilabel datasets:

- **birds** - Bird species classification from audio features
- **bookmarks** - Web bookmark categorization  
- **Cal500** - Music emotion classification
- **corel5k** - Image annotation with Corel 5K dataset
- **delicious** - Social bookmarking tag prediction
- **Emotions** - Music emotion classification
- **enron** - Email classification
- **genbase** - Gene functional classification
- **mediamill** - Video semantic annotation
- **yeast** - Yeast protein functional classification

## 🛠️ Installation

1. **Clone and setup:**
   ```bash
   git clone <your-repo-url>
   cd Multillabel-Datasets
   git submodule update --init --recursive
   python setup.py
   ```

2. **Quick start:**
   ```bash
   python quickstart.py
   ```

## 📚 Available Scripts

### 🎯 Interactive Training
```bash
python multilabel_trainer.py --interactive
```
- User-friendly dataset selection
- Automatic feature/label detection
- Hyperparameter optimization with Bayesian search
- Comprehensive evaluation and reporting

### 🔮 Model Inference
```bash
python lift_inference.py --interactive
```
- Load trained models
- Make predictions on new data
- Batch inference from CSV files
- Model evaluation on test data

### 🔍 Dataset Explorer
```bash
python dataset_explorer.py --interactive
```
- Analyze dataset characteristics
- Generate statistical reports
- Compare multiple datasets
- Export detailed HTML/JSON reports

### 📊 Batch Runner
```bash
python batch_runner.py --comparison
```
- Train models on multiple datasets
- Compare optimization strategies
- Automated experiment logging

### 🏃 Simple Example
```bash
python run_lift_experiment.py --dataset yeast
```
- Basic LIFT training example
- Good for understanding the workflow

## 🔧 Command Line Usage

### Train a specific dataset:
```bash
python multilabel_trainer.py --dataset yeast --optimize
```

### Make predictions:
```bash
python lift_inference.py --model trained_models/yeast_model.pkl --data new_data.csv
```

### Analyze all datasets:
```bash
python dataset_explorer.py --all
```

### Run batch experiments:
```bash
python batch_runner.py --datasets yeast emotions birds
```

## 📊 Features

- **Automatic Dataset Processing**: Smart detection of features and labels
- **Hyperparameter Optimization**: Bayesian optimization for best performance
- **Comprehensive Evaluation**: Multiple multilabel metrics and per-label analysis
- **Interactive Interface**: User-friendly menu-driven operations
- **Batch Processing**: Train multiple datasets automatically
- **Model Persistence**: Save and load trained models
- **Rich Reporting**: HTML and JSON reports with visualizations
- **Dataset Analysis**: Statistical analysis and comparison tools

## 📁 Project Structure

```
Multilabel-Datasets/
├── quickstart.py              # Interactive quick start interface
├── multilabel_trainer.py      # Main training system
├── lift_inference.py          # Model inference and evaluation
├── dataset_explorer.py        # Dataset analysis tools
├── batch_runner.py            # Batch experiment runner
├── run_lift_experiment.py     # Simple example script
├── setup.py                   # Installation script
├── requirements.txt           # Python dependencies
├── USAGE_GUIDE.md             # Detailed documentation
├── *.zip                      # Dataset files
├── LIFT-MultiLabel-Learning-with-Label-Specific-Features/  # LIFT submodule
└── [Generated directories]
    ├── extracted_datasets/    # Extracted dataset files
    ├── trained_models/        # Saved models
    ├── reports/              # Training reports
    ├── dataset_reports/      # Dataset analysis reports
    └── predictions/          # Inference outputs
```

## 📈 Example Workflow

1. **Explore datasets:**
   ```bash
   python dataset_explorer.py --dataset yeast
   ```

2. **Train a model:**
   ```bash
   python multilabel_trainer.py --dataset yeast --optimize
   ```

3. **Make predictions:**
   ```bash
   python lift_inference.py --interactive
   ```

4. **Compare multiple datasets:**
   ```bash
   python batch_runner.py --comparison
   ```

## 🎯 Evaluation Metrics

The system provides comprehensive multilabel evaluation:

- **Overall Metrics**: Hamming Loss, Jaccard Score, F1 Scores (micro/macro/weighted)
- **Per-Label Metrics**: Precision, Recall, F1 Score, Support
- **Advanced Analysis**: Label cardinality, density, frequency distributions

## 📖 Documentation

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project follows the same license as the LIFT package (MIT).

---

**Get started now:** `python quickstart.py` 🚀

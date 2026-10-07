# Contributing to Multi-label Datasets with LIFT Training System

We welcome contributions to improve the dataset collection, training scripts, and evaluation tools!

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally (with submodules):
   ```bash
   git clone --recurse-submodules https://github.com/your-username/Multillabel-Datasets.git
   cd Multillabel-Datasets
   ```

3. **Set up development environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ./LIFT-MultiLabel-Learning-with-Label-Specific-Features
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** and test them

6. **Submit a pull request**

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git (with submodule support)
- Jupyter Notebook (optional)

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./LIFT-MultiLabel-Learning-with-Label-Specific-Features
pip install black isort flake8 pytest
```

## 📁 Project Structure

```
Multilabel-Datasets/
├── quickstart.py              # Interactive quick start
├── multilabel_trainer.py      # Main training system
├── lift_inference.py          # Model inference
├── dataset_explorer.py        # Dataset analysis
├── batch_runner.py            # Batch experiments
├── run_lift_experiment.py     # Simple example
├── setup.py                   # Installation
├── requirements.txt           # Dependencies
├── USAGE_GUIDE.md             # Documentation
├── *.zip                      # Dataset archives
├── LIFT-MultiLabel-Learning-with-Label-Specific-Features/  # Submodule
├── extracted_datasets/        # Extracted data (gitignored)
├── trained_models/            # Saved models (gitignored)
├── reports/                   # Training reports (gitignored)
├── dataset_reports/           # Analysis reports (gitignored)
└── predictions/               # Inference outputs (gitignored)
```

## 📝 Code Style

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- Line length: 88 characters (Black default)

```bash
# Format
black .

# Sort imports
isort .

# Check
flake8 . --max-line-length=88 --extend-ignore=E203,W503 --exclude=.git,__pycache__,.venv,venv,env,extracted_datasets,LIFT-MultiLabel-Learning-with-Label-Specific-Features
```

## 🧪 Testing Guidelines

### Test Categories
1. **Script Execution**: All CLI scripts should run without errors
2. **Dataset Loading**: Datasets should load correctly
3. **Training Pipeline**: End-to-end training should work
4. **Inference**: Model loading and prediction should work

### Manual Testing
```bash
# Test quickstart
python quickstart.py --help

# Test dataset explorer
python dataset_explorer.py --help

# Test trainer
python multilabel_trainer.py --help

# Test inference
python lift_inference.py --help
```

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines (`black`, `isort`, `flake8` pass)
- [ ] Scripts execute without errors (`--help` works)
- [ ] New functionality includes documentation
- [ ] README.md is updated if needed
- [ ] Submodule changes are coordinated

### Pull Request Description
Include:
- **Purpose**: What does this PR accomplish?
- **Changes**: What specific changes were made?
- **Testing**: How was this tested?
- **Datasets**: Which datasets were tested (if applicable)?

### Example PR Description
```
## Purpose
Add support for new multi-label dataset 'mediamill'

## Changes
- Added mediamill dataset download and extraction
- Updated dataset_explorer.py with mediamill statistics
- Added mediamill to batch_runner.py comparison
- Documented dataset characteristics in USAGE_GUIDE.md

## Testing
- Verified dataset loads correctly
- Trained LIFT model on mediamill (F1: 0.52)
- Added to batch comparison

## Breaking Changes
None
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, package versions
2. **Script**: Which script has the issue
3. **Dataset**: Which dataset (if applicable)
4. **Error Messages**: Full error traceback

## 💡 Feature Requests

For new features:
1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this needed?
3. **Consider dataset compatibility** - works with all 10 datasets?
4. **Propose implementation** if you have ideas

## 📊 Dataset Contributions

To add a new dataset:
1. Ensure it's a standard multi-label benchmark
2. Provide download script or source URL
3. Document: samples, features, labels, domain
4. Add to dataset_explorer.py statistics
5. Test with LIFT training pipeline
6. Update USAGE_GUIDE.md and README.md

## 📚 Documentation

- Update README.md for user-facing changes
- Update USAGE_GUIDE.md for detailed usage
- Add docstrings for new functions
- Include examples for new CLI options

## 🤝 Community Guidelines

- **Be respectful** and inclusive
- **Help others** reproduce results
- **Share benchmarks** - comparison is valuable
- **Ask questions** if something is unclear
- **Provide constructive feedback**

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion

Thank you for contributing to Multi-label Datasets! 🎉
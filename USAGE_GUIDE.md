# Multilabel Dataset Training System with LIFT

A comprehensive system for training, evaluating, and deploying multilabel classifiers using the LIFT (Learning with Label-Specific Features) algorithm on various multilabel datasets.

## 🚀 Features

- **Interactive Training**: User-friendly interface for dataset selection and model training
- **Hyperparameter Optimization**: Bayesian optimization for automatic hyperparameter tuning  
- **Model Inference**: Load trained models and make predictions on new data
- **Dataset Exploration**: Comprehensive analysis and comparison of multilabel datasets
- **Automated Reporting**: Generate detailed HTML and JSON reports
- **Batch Processing**: Support for both interactive and command-line batch processing

## 📦 Available Datasets

The system supports the following multilabel datasets:

1. **birds** - Bird species classification from audio features
2. **bookmarks** - Web bookmark categorization
3. **Cal500** - Music emotion classification (CAL500)
4. **corel5k** - Image annotation with Corel 5K dataset
5. **delicious** - Social bookmarking tag prediction
6. **Emotions** - Music emotion classification
7. **enron** - Email classification (Enron dataset)
8. **genbase** - Gene functional classification
9. **mediamill** - Video semantic annotation
10. **yeast** - Yeast protein functional classification

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Multillabel-Datasets
```

### 2. Initialize LIFT Submodule
```bash
git submodule update --init --recursive
```

### 3. Run Setup Script
```bash
python setup.py
```

This will:
- Check Python version compatibility (3.8+)
- Install required dependencies
- Set up the LIFT package
- Create necessary directories
- Verify the installation

### 4. Manual Installation (Alternative)
```bash
pip install -r requirements.txt
cd LIFT-MultiLabel-Learning-with-Label-Specific-Features
pip install -e .
cd ..
```

## 📚 Usage

### Interactive Training System

Start the interactive training system:
```bash
python multilabel_trainer.py --interactive
```

Or train a specific dataset:
```bash
python multilabel_trainer.py --dataset yeast --optimize
```

The training system will:
1. Show available datasets
2. Extract and load the selected dataset
3. Automatically detect features and labels
4. Perform train/test split if needed
5. Scale features using StandardScaler
6. Train LIFT model with optional Bayesian optimization
7. Evaluate model performance
8. Save trained model and generate reports

### Model Inference

Use trained models for predictions:
```bash
python lift_inference.py --interactive
```

Batch inference on a CSV file:
```bash
python lift_inference.py --model trained_models/yeast_model.pkl --data new_data.csv --output predictions.csv
```

Evaluate a model on test data:
```bash
python lift_inference.py --model trained_models/yeast_model.pkl --data test_data.csv --evaluate
```

### Dataset Exploration

Explore and analyze datasets:
```bash
python dataset_explorer.py --interactive
```

Analyze a specific dataset:
```bash
python dataset_explorer.py --dataset yeast
```

Compare multiple datasets:
```bash
python dataset_explorer.py --compare yeast emotions birds
```

Analyze all available datasets:
```bash
python dataset_explorer.py --all
```

## 📁 Project Structure

```
Multillabel-Datasets/
├── multilabel_trainer.py      # Main training script
├── lift_inference.py          # Model inference script
├── dataset_explorer.py        # Dataset analysis script
├── setup.py                   # Installation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── *.zip                      # Dataset archive files
├── LIFT-MultiLabel-Learning-with-Label-Specific-Features/  # LIFT package submodule
├── extracted_datasets/        # Extracted dataset files
├── trained_models/           # Saved trained models
├── reports/                  # Training reports (JSON/HTML)
├── dataset_reports/          # Dataset analysis reports
└── predictions/              # Inference outputs
```

## 🔧 Configuration and Customization

### Training Parameters

The training system uses the following default parameters:
- **k**: Number of clusters per label (optimized via Bayesian search)
- **Base Classifier**: LogisticRegression (can be customized)
- **Optimization**: 15 iterations of Bayesian optimization
- **Cross-validation**: 3-fold CV for hyperparameter tuning
- **Test Split**: 20% for validation when no separate test set is available

### Feature Detection

The system automatically detects features and labels using heuristics:
1. Identifies binary columns (0/1 values) as potential labels
2. If >50% of columns are binary, assumes they are labels
3. Otherwise, assumes last few columns are labels
4. Remaining columns are treated as features

### Customizing Base Classifiers

You can modify the LIFT classifier to use different base estimators:

```python
from sklearn.ensemble import RandomForestClassifier
from lift_ml import LIFTClassifier

# Custom classifier
clf = LIFTClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    k=3,
    random_state=42
)
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Overall Metrics
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **Jaccard Score**: Intersection over union of predicted and true labels
- **F1 Scores**: Micro, macro, and weighted averages
- **Subset Accuracy**: Exact match accuracy for complete label sets

### Per-Label Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1 Score**: Harmonic mean of precision and recall
- **Support**: Number of true instances for each label

## 📈 Reports and Output

### Training Reports
- **JSON Report**: Machine-readable metrics and metadata
- **HTML Report**: Human-readable formatted report with tables
- **Model Files**: Serialized models with metadata (`.pkl` files)

### Dataset Analysis Reports
- **Dataset Statistics**: Samples, features, labels, file information
- **Label Analysis**: Cardinality, density, frequency distributions
- **Comparison Reports**: Multi-dataset comparisons and rankings

### Inference Output
- **Prediction CSV**: Original data + prediction columns
- **Probability Scores**: Class probabilities (when available)
- **Evaluation Reports**: Performance metrics on test data

## 🐛 Troubleshooting

### Common Issues

1. **LIFT package not found**
   ```bash
   # Reinitialize submodule
   git submodule update --init --recursive
   cd LIFT-MultiLabel-Learning-with-Label-Specific-Features
   pip install -e .
   ```

2. **Dataset extraction fails**
   - Ensure dataset .zip files are present in the root directory
   - Check file permissions and available disk space

3. **Memory issues with large datasets**
   - Use smaller datasets for initial testing
   - Consider using sampling for very large datasets
   - Monitor memory usage during training

4. **Feature/label detection issues**
   - Manually inspect CSV files to understand structure
   - Modify detection heuristics if needed
   - Ensure consistent column naming across train/test splits

### Performance Tips

1. **Speed up training**
   - Reduce Bayesian optimization iterations (`n_iter`)
   - Use smaller k values for clustering
   - Disable hyperparameter optimization for quick testing

2. **Improve accuracy**
   - Enable Bayesian optimization
   - Increase the search space for hyperparameters
   - Use cross-validation for model selection
   - Try different base estimators

3. **Handle imbalanced labels**
   - LIFT inherently handles label imbalance through clustering
   - Consider stratified sampling for train/test splits
   - Monitor per-label metrics for rare labels

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project uses the same license as the LIFT package (MIT License).

## 🙏 Acknowledgments

- **LIFT Algorithm**: Based on the paper "Learning with Label-Specific Features"
- **Datasets**: Various sources - see individual dataset documentation
- **Dependencies**: Scikit-learn, NumPy, Pandas, and other open-source libraries

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the script help messages (`--help` flag)
3. Examine the generated reports for debugging information
4. Create an issue in the repository

---

Happy multilabel learning! 🎯

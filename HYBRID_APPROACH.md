# Hybrid Melanoma Detection System

## Overview

This hybrid system combines **two complementary approaches** to achieve >90% precision, recall, and accuracy:

1. **Quantum Neural Network (QNN)** - Processes raw images directly
2. **Quantum Support Vector Classifier (QSVC)** - Processes ResNet50 embeddings

The system uses **SMOTE balancing** and **weighted ensemble voting** to combine predictions.

## Architecture

```
Input Data
    │
    ├─→ Raw Images ──→ QNN (Quantum Neural Network) ──→ Predictions
    │
    └─→ Images ──→ ResNet50 ──→ Embeddings ──→ QSVC ──→ Predictions
                                                          │
                                                          ↓
                                              Weighted Ensemble Voting
                                                          ↓
                                                   Final Predictions
```

## Key Components

### 1. Quantum Neural Network (QNN)

- **Input**: Raw images (224x224x3)
- **Architecture**:
  - ResNet18 feature extractor (fine-tuned)
  - Quantum variational circuit (6 qubits, 3 layers)
  - Classical classifier
- **Training**:
  - Focal loss for imbalanced data
  - Weighted sampling (50/50 balance)
  - 60 epochs with early stopping

### 2. Quantum Support Vector Classifier (QSVC)

- **Input**: ResNet50 embeddings (2048-dim)
- **Processing**:
  - SMOTE balancing (SMOTETomek)
  - Robust normalization
  - PCA reduction (2048 → 12 dimensions)
  - Hybrid quantum-classical kernel
- **Training**:
  - 1000 balanced samples (with SMOTE)
  - Grid search for optimal C
  - 5-fold cross-validation

### 3. SMOTE Balancing

- **SMOTETomek**: Combines SMOTE oversampling with Tomek links undersampling
- Applied to both training and validation sets
- Creates perfectly balanced datasets

### 4. Ensemble Voting

- **Weighted Voting**: Combines predictions based on individual model performance
- **Weight Calculation**: Based on F1 scores, with bonus for >90% models
- **Threshold Optimization**: Finds optimal threshold for ensemble probabilities

## Performance Improvements

### Why This Approach Works Better

1. **Complementary Models**:

   - QNN captures spatial patterns in raw images
   - QSVC captures high-level semantic features from embeddings
   - Ensemble combines both perspectives

2. **SMOTE Balancing**:

   - Addresses severe class imbalance (11% vs 89%)
   - Creates synthetic minority samples
   - Prevents model bias toward majority class

3. **Weighted Ensemble**:

   - Automatically weights better-performing models
   - Adapts to different data distributions
   - More robust than single models

4. **Optimized Hyperparameters**:
   - More training samples (1000 vs 600)
   - Better PCA retention (12 vs 8 components)
   - Wider hyperparameter search

## Expected Performance

With this hybrid approach:

- **Precision**: >90% (target)
- **Recall**: >90% (target)
- **Balanced Accuracy**: >90% (target)
- **F1-Score**: >0.90
- **AUC-ROC**: >0.95

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run hybrid pipeline
python main.py
```

## Pipeline Steps

1. **Data Loading**: Patient-level split (no data leakage)
2. **Embedding Extraction**: ResNet50 features (cached if exists)
3. **QNN Training**: Image-based quantum neural network
4. **QSVC Training**: Embedding-based quantum SVM (with SMOTE)
5. **Ensemble Evaluation**: Weighted voting on test set
6. **Results**: Comprehensive metrics and comparison

## Configuration

Key parameters in `main.py`:

```python
# QNN Configuration
num_epochs=60
batch_size=32
lr=0.0005
use_focal_loss=True

# QSVC Configuration
n_train_samples=1000  # With SMOTE
n_val_samples=500
n_pca_components=12
use_hybrid=True
quantum_weight=0.5

# Ensemble
ensemble_method='weighted_voting'
use_smote=True
```

## Output Files

- `results/hybrid_results.json`: Complete results with individual and ensemble metrics
- `embeddings/*.csv`: Cached ResNet50 embeddings
- Model checkpoints (if saved)

## Troubleshooting

### Low Performance

- Increase training samples
- Adjust ensemble weights manually
- Try different SMOTE strategies
- Increase PCA components

### Memory Issues

- Reduce batch size
- Use fewer training samples
- Process in chunks

### Slow Training

- Use GPU for QNN training
- Reduce number of epochs
- Use cached embeddings

## References

- SMOTE: Synthetic Minority Oversampling Technique
- Quantum Neural Networks with PennyLane
- Quantum Support Vector Machines
- Ensemble Learning Methods

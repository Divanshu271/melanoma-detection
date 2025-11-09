# Quantum Machine Learning for Melanoma Detection - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Architecture & Methodology](#architecture--methodology)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture](#model-architecture)
7. [Training Process](#training-process)
8. [Key Features & Techniques](#key-features--techniques)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results & Performance](#results--performance)
11. [Usage Instructions](#usage-instructions)
12. [File Structure](#file-structure)
13. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a **Hybrid Quantum-Classical Neural Network** for binary classification of melanoma vs non-melanoma skin lesions using the HAM10000 dataset. The model combines classical deep learning (ResNet18) with quantum neural networks (QNN) using PennyLane for quantum computation.

### Key Objectives:

- Achieve **>70% precision and recall** for both melanoma and non-melanoma classification
- Prevent **data leakage** through proper patient-level splitting
- Implement a **robust QML pipeline** with classical feature extraction
- Handle **severe class imbalance** in the dataset
- Optimize for **balanced performance** across all metrics

### Technology Stack:

- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning framework
- **scikit-learn**: Data preprocessing and metrics
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

---

## Problem Statement

### Challenges:

1. **Data Leakage**: Original papers showed 99% accuracy, likely due to data leakage from improper train/test splits
2. **Class Imbalance**: HAM10000 has significantly more non-melanoma samples than melanoma samples
3. **Inconsistent Dataset**: HAM10000 is noted to be "very inconsistent" with quality variations
4. **High Precision & Recall**: Need both metrics >70% (not just accuracy) for medical applications

### Solution Approach:

- **Patient-level splitting** to ensure no data leakage
- **Advanced class balancing** techniques (weighted sampling, class weights, focal loss)
- **Precision-focused optimization** to balance metrics
- **Robust regularization** to prevent overfitting

---

## Dataset Description

### HAM10000 Dataset:

- **Source**: Human Against Machine with 10000 training images
- **Classes**: 7 skin lesion types (converted to binary: melanoma vs non-melanoma)
- **Images**: Approximately 10,000 dermoscopic images
- **Distribution**: Highly imbalanced (melanoma is minority class)
- **Metadata**: Includes lesion_id, image_id, dx (diagnosis), and other clinical data

### Data Structure:

archive/
├── HAM10000_metadata.csv # Metadata with labels and patient info
├── HAM10000_images_part_1/ # Image directory 1
└── HAM10000_images_part_2/ # Image directory 2

### Key Columns in Metadata:

- `image_id`: Unique identifier for each image
- `lesion_id`: Unique identifier for each lesion (used for patient-level splitting)
- `dx`: Diagnosis (mel = melanoma, others = non-melanoma)
- `is_melanoma`: Binary label (1 = melanoma, 0 = non-melanoma)

---

## Architecture & Methodology

### Hybrid Quantum-Classical Architecture:

Input Image (224x224x3)
↓
ResNet18 Feature Extractor (Frozen/Fine-tuned)
↓
Feature Vector (512-dim)
↓
Dimensionality Reduction (512 → 16 for quantum encoding)
↓
Quantum Feature Map (Amplitude Encoding)
↓
Variational Quantum Circuit (4 qubits, 2 layers)
↓
Quantum Measurements (4 expectation values)
↓
Classical Classifier (128 → 64 → 2)
↓
Output: Melanoma / Non-Melanoma

### Quantum Component:

- **Qubits**: 4 qubits
- **Layers**: 2 variational layers
- **Feature Map**: Amplitude encoding
- **Variational Circuit**:
  - Rotation gates (RY, RZ) per qubit
  - Entangling gates (CNOT) for qubit coupling
- **Measurements**: Pauli-Z expectation values

### Classical Component:

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Feature Extraction**: Final convolutional layers
- **Classifier**: Fully connected layers with BatchNorm and Dropout

---

## Data Preprocessing

### 1. Data Loading (`src/data_preprocessing.py`)

#### Patient-Level Splitting:

```python
# Key: Split by lesion_id, not image_id
# This prevents data leakage from same patient appearing in multiple sets
unique_lesions = metadata['lesion_id'].unique()
train_lesions, val_lesions, test_lesions = stratified_split(unique_lesions)
```

**Why Patient-Level?**

- Same lesion can have multiple images
- Same patient can have multiple lesions
- Prevents information leakage between train/val/test

#### Data Leakage Prevention:

- ✅ No overlap of `lesion_id` between train/val/test
- ✅ Separate preprocessing pipelines for each split
- ✅ Assertions to verify no leakage

### 2. Data Augmentation (`src/data_loader.py`)

#### Training Augmentation:

- Resize: 224×224
- Random Horizontal Flip: p=0.5
- Random Rotation: ±20 degrees
- Color Jitter: Brightness, contrast, saturation
- Random Erasing: p=0.2

#### Validation/Test:

- Resize: 224×224
- **NO augmentation** (evaluation only)

### 3. Class Balancing:

#### Weighted Random Sampling:

- 50% melanoma samples, 50% non-melanoma samples per batch
- Oversamples minority class (melanoma) during training

#### Class Weights for Loss:

- Calculated as inverse frequency
- Capped at 2.5x ratio to prevent overcorrection
- Applied to CrossEntropyLoss or FocalLoss

---

## Model Architecture

### 1. Quantum Neural Network (`src/quantum_models.py`)

#### QuantumFeatureMap Class:

```python
- Amplitude encoding of classical features
- Normalizes features to valid quantum state
- Resizes to 2^n_qubits dimensions
```

#### QuantumNeuralNetwork Class:

**Components:**

1. **Feature Extractor**: ResNet18 (pretrained, optionally fine-tuned)
2. **Feature Reduction**: Linear layer (512 → 16)
3. **Quantum Circuit**: PennyLane QNode with:
   - Amplitude embedding
   - Variational layers (RY, RZ rotations + CNOT entangling)
   - Pauli-Z measurements
4. **Classifier**: 3-layer MLP (128 → 64 → 2) with:
   - BatchNorm
   - ReLU activation
   - Dropout (0.6, 0.5) for regularization

**Forward Pass:**

```python
1. Extract features with ResNet18
2. Reduce dimensions for quantum encoding
3. Normalize to quantum state
4. Apply quantum circuit (amplitude encoding + VQC)
5. Measure expectation values
6. Classical classification on quantum outputs
```

#### Fine-Tuning Strategy:

- **Feature Extractor**: Can be frozen (default) or fine-tuned
- **Fine-tuning LR**: 0.1× base learning rate
- **New Layers LR**: Full learning rate

### 2. Classical Baseline (`src/quantum_models.py`)

For comparison:

- Same ResNet18 backbone
- Simple MLP classifier
- No quantum component

---

## Training Process

### 1. Loss Functions (`src/training.py`)

#### Weighted CrossEntropyLoss:

```python
loss = CrossEntropyLoss(weight=class_weights)
# class_weights = [weight_non_mel, weight_mel]
```

#### Focal Loss (Optional):

```python
FocalLoss(alpha=1, gamma=1.5, weight=class_weights)
# Focuses on hard examples
# Reduces impact of easy negative samples
```

### 2. Optimization (`src/training.py`)

#### Optimizer:

- **Algorithm**: Adam
- **Learning Rate**: 0.001 (new layers), 0.0001 (fine-tuning)
- **Weight Decay**: 1e-3 (L2 regularization)
- **Gradient Clipping**: max_norm=1.0

#### Learning Rate Scheduling:

- **Scheduler**: ReduceLROnPlateau
- **Mode**: min (monitor validation loss)
- **Factor**: 0.5
- **Patience**: 5 epochs

### 3. Threshold Optimization

#### Dynamic Threshold Finding:

```python
find_optimal_threshold(y_true, y_probs, min_precision=0.70, min_recall=0.70)
```

**Strategy:**

1. Search thresholds from 0.1 to 0.95 (step 0.02)
2. **Priority 1**: Find threshold where both precision AND recall ≥ 70%
3. **Priority 2**: If none found, prioritize precision with 2x bonus
4. **Scoring**: Harmonic mean of precision and recall with bonuses

**Why Dynamic Threshold?**

- Default 0.5 threshold doesn't account for class imbalance
- Optimizes for specific precision/recall targets
- Balances false positives and false negatives

### 4. Model Selection Criteria

**Composite Score Calculation:**

```python
if precision >= 0.70 and recall >= 0.70:
    score = f1 * 3.0  # Massive bonus for meeting targets
elif precision >= 0.70 and recall >= 0.65:
    score = f1 * 2.5  # Precision priority
elif precision >= 0.65 and recall >= 0.70:
    score = f1 * 2.0  # Lower than above
# ... plus accuracy and balanced accuracy bonuses
```

**Saves best model based on:**

- F1-score (base)
- Precision & Recall targets (>70%)
- Balanced accuracy
- Penalty for severe overfitting

### 5. Early Stopping

- **Patience**: 8 epochs (default)
- **Monitor**: Validation score
- **Early Exit**: If targets met (precision & recall >70%) and no improvement for 5 epochs

### 6. Regularization Techniques

#### To Prevent Overfitting:

1. **Dropout**: 0.6 in first layer, 0.5 in second layer
2. **Weight Decay**: 1e-3 (L2 regularization)
3. **Gradient Clipping**: Prevents exploding gradients
4. **Batch Normalization**: Stabilizes training
5. **Data Augmentation**: Increases dataset diversity

---

## Key Features & Techniques

### 1. Data Leakage Prevention ✅

**Problem**: Same patient/lesion appearing in multiple splits

**Solution**:

- Split by `lesion_id` instead of `image_id`
- Stratified splitting to maintain class distribution
- Assertions to verify no overlap:
  ```python
  assert len(train_lesions & val_lesions) == 0
  assert len(train_lesions & test_lesions) == 0
  assert len(val_lesions & test_lesions) == 0
  ```

### 2. Class Imbalance Handling

**Problem**: Melanoma is minority class (~11% of data)

**Solutions**:

1. **Weighted Random Sampling**: 50/50 split per batch
2. **Class Weights**: 2.5x ratio for loss function
3. **Focal Loss**: Focuses on hard examples
4. **Threshold Optimization**: Balances precision/recall

### 3. Precision-Focused Optimization

**Problem**: Model overcorrects (high recall, low precision ~40%)

**Solutions**:

1. **Precision-Priority Threshold Search**: 2x bonus for precision ≥70%
2. **Model Selection**: Heavily weights precision in scoring
3. **Regularization**: Prevents overfitting that causes low precision
4. **Balanced Sampling**: 50/50 instead of aggressive oversampling

### 4. Quantum Circuit Optimization

**Challenges**:

- Quantum circuits process samples sequentially (slow)
- Memory limitations for large batches

**Solutions**:

- Chunked processing for batches >16
- Efficient tensor operations
- Device management (CPU/GPU)

### 5. SSL Certificate Handling

**Issue**: macOS SSL certificate verification errors when downloading pretrained weights

**Solution**:

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

## Evaluation Metrics

### Primary Metrics:

1. **Accuracy**: Overall correctness
   Why Patient-Level?
   Same lesion can have multiple images
   Same patient can have multiple lesions
   Prevents information leakage between train/val/test
   Data Leakage Prevention:
   ✅ No overlap of lesion_id between train/val/test
   ✅ Separate preprocessing pipelines for each split
   ✅ Assertions to verify no leakage
2. Data Augmentation (src/data_loader.py)
   Training Augmentation:
   Resize: 224×224
   Random Horizontal Flip: p=0.5
   Random Rotation: ±20 degrees
   Color Jitter: Brightness, contrast, saturation
   Random Erasing: p=0.2
   Validation/Test:
   Resize: 224×224
   NO augmentation (evaluation only)
3. Class Balancing:
   Weighted Random Sampling:
   50% melanoma samples, 50% non-melanoma samples per batch
   Oversamples minority class (melanoma) during training
   Class Weights for Loss:
   Calculated as inverse frequency
   Capped at 2.5x ratio to prevent overcorrection
   Applied to CrossEntropyLoss or FocalLoss
   Model Architecture
4. Quantum Neural Network (src/quantum_models.py)
   QuantumFeatureMap Class:

- Amplitude encoding of classical features
- Normalizes features to valid quantum state
- Resizes to 2^n_qubits dimensions
  QuantumNeuralNetwork Class:
  Components:
  Feature Extractor: ResNet18 (pretrained, optionally fine-tuned)
  Feature Reduction: Linear layer (512 → 16)
  Quantum Circuit: PennyLane QNode with:
  Amplitude embedding
  Variational layers (RY, RZ rotations + CNOT entangling)
  Pauli-Z measurements
  Classifier: 3-layer MLP (128 → 64 → 2) with:
  BatchNorm
  ReLU activation
  Dropout (0.6, 0.5) for regularization
  Forward Pass:

1. Extract features with ResNet18
2. Reduce dimensions for quantum encoding
3. Normalize to quantum state
4. Apply quantum circuit (amplitude encoding + VQC)
5. Measure expectation values
6. Classical classification on quantum outputs
   Fine-Tuning Strategy:
   Feature Extractor: Can be frozen (default) or fine-tuned
   Fine-tuning LR: 0.1× base learning rate
   New Layers LR: Full learning rate
7. Classical Baseline (src/quantum_models.py)
   For comparison:
   Same ResNet18 backbone
   Simple MLP classifier
   No quantum component
   Training Process
8. Loss Functions (src/training.py)
   Weighted CrossEntropyLoss:
   loss = CrossEntropyLoss(weight=class_weights)

# class_weights = [weight_non_mel, weight_mel]

Focal Loss (Optional):
FocalLoss(alpha=1, gamma=1.5, weight=class*weights)# Focuses on hard examples# Reduces impact of easy negative samples 2. Optimization (src/training.py)
Optimizer:
Algorithm: Adam
Learning Rate: 0.001 (new layers), 0.0001 (fine-tuning)
Weight Decay: 1e-3 (L2 regularization)
Gradient Clipping: max_norm=1.0
Learning Rate Scheduling:
Scheduler: ReduceLROnPlateau
Mode: min (monitor validation loss)
Factor: 0.5
Patience: 5 epochs 3. Threshold Optimization
Dynamic Threshold Finding:
find_optimal_threshold(y_true, y_probs, min_precision=0.70, min_recall=0.70)
Strategy:
Search thresholds from 0.1 to 0.95 (step 0.02)
Priority 1: Find threshold where both precision AND recall ≥ 70%
Priority 2: If none found, prioritize precision with 2x bonus
Scoring: Harmonic mean of precision and recall with bonuses
Why Dynamic Threshold?
Default 0.5 threshold doesn't account for class imbalance
Optimizes for specific precision/recall targets
Balances false positives and false negatives 4. Model Selection Criteria
Composite Score Calculation:
if precision >= 0.70 and recall >= 0.70: score = f1 * 3.0 # Massive bonus for meeting targetselif precision >= 0.70 and recall >= 0.65: score = f1 _ 2.5 # Precision priorityelif precision >= 0.65 and recall >= 0.70: score = f1 _ 2.0 # Lower than above# ... plus accuracy and balanced accuracy bonusesel and optimal threshold from validation**Important**: All metrics use the **optimal threshold** (not default 0.5)---## Results & Performance### Expected Performance:#### After Optimizations:- **Precision**: 60-75% (improved from ~40%)- **Recall**: 70-80% (maintained)- **Accuracy**: 80-88%- **F1-Score**: 0.65-0.75- **Balanced Accuracy**: 75-82%#### Target Achievement:- ✅ **Recall > 70%**: Achieved- ⚠️ **Precision > 70%**: Improved, working toward target- ✅ **Accuracy**: High (80%+)- ✅ **No Data Leakage**: Verified through assertions### Overfitting Management:**Before Optimization:**- Train Precision: 92%- Val Precision: 39%- Gap: 53% (severe overfitting)**After Optimization:**- Train Precision: ~90%- Val Precision: 60-75%- Gap: 15-30% (acceptable)### Training Time:- **Per Epoch**: ~8-10 minutes (CPU) / ~3-5 minutes (GPU)- **Total Training**: ~40 epochs = 5-6 hours (CPU) / 2-3 hours (GPU)- **Quantum Processing**: ~2-3 seconds per batch---## Usage Instructions### 1. Installation`bash# Install dependenciespip install -r requirements.txt`**Requirements:**- torch >= 2.0.0- torchvision >= 0.15.0- pennylane >= 0.32.0- numpy >= 1.24.0- pandas >= 2.0.0- scikit-learn >= 1.3.0- matplotlib >= 3.7.0- seaborn >= 0.12.0- Pillow >= 10.0.0- tqdm >= 4.65.0### 2. Dataset Setup`Zinta/├── archive/│   ├── HAM10000_metadata.csv│   ├── HAM10000_images_part_1/│   └── HAM10000_images_part_2/├── src/├── results/└── main.py`### 3. Training`bash# Train QML model with default settingspython main.py`**Configuration (in `main.py`):**`python- num_epochs: 40- learning_rate: 0.001- batch_size: 32- patience: 8- use_focal_loss: False (uses Weighted CrossEntropy)`### 4. Custom TrainingModify `main.py`:`pythontrainer.train(    train_loader, val_loader,    num_epochs=50,      # Change epochs    lr=0.0005,          # Change learning rate    patience=10,         # Change patience    use_focal_loss=True  # Use Focal Loss instead)`### 5. EvaluationAfter training, the model automatically:- Evaluates on test set- Saves training history plot- Prints comprehensive metrics---## File Structure`Zinta/│├── main.py                          # Main training script├── requirements.txt                  # Python dependencies├── README.md                         # Quick start guide├── PROJECT_DOCUMENTATION.md          # This file│├── src/│   ├── data_preprocessing.py         # Data loading & patient-level splitting│   ├── data_loader.py                # Dataset class & data loaders│   ├── quantum_models.py             # QNN & Classical baseline models│   └── training.py                   # Trainer class & evaluation│├── archive/                          # Dataset (not in repo)│   ├── HAM10000_metadata.csv│   ├── HAM10000_images_part_1/│   └── HAM10000_images_part_2/│└── results/                          # Output directory    └── training_history.png          # Training curves`### File Descriptions:#### `main.py`- Orchestrates entire workflow- Loads data, initializes model, trains, evaluates#### `src/data_preprocessing.py`- `HAM10000DataLoader` class- Patient-level splitting- Data leakage prevention#### `src/data_loader.py`- `MelanomaDataset` PyTorch dataset- Data augmentation- Class balancing (weighted sampling)- Class weights calculation#### `src/quantum_models.py`- `QuantumNeuralNetwork`: Hybrid QML model- `ClassicalBaseline`: Classical comparison- `QuantumFeatureMap`: Quantum encoding- ResNet18 loading with SSL handling#### `src/training.py`- `Trainer` class: Training loop, validation, evaluation- `FocalLoss`: Alternative loss function- Threshold optimization- Model selection- Metrics calculation- Visualization---## Future Improvements### 1. Model Architecture- [ ] Experiment with more qubits (6, 8)- [ ] Deeper quantum circuits (3-4 layers)- [ ] Different quantum feature maps (angle encoding)- [ ] Quantum data re-uploading- [ ] Hybrid quantum-classical layers### 2. Training Optimization- [ ] Learning rate scheduling (cosine annealing)- [ ] Ensemble methods (multiple QNNs)- [ ] Transfer learning from other medical datasets- [ ] Active learning for annotation### 3. Data Improvements- [ ] Additional data augmentation (mixup, cutmix)- [ ] Synthetic data generation (GANs)- [ ] Multi-scale training- [ ] Test-time augmentation### 4. Evaluation- [ ] Cross-validation- [ ] Stratified k-fold evaluation- [ ] Bootstrapping for confidence intervals- [ ] ROC curve analysis- [ ] Precision-Recall curve analysis### 5. Performance- [ ] Quantum hardware execution (real quantum computers)- [ ] Model quantization- [ ] Knowledge distillation- [ ] Faster quantum simulation### 6. Medical Applications- [ ] Integration with DICOM format- [ ] Real-time inference API- [ ] Clinical decision support system- [ ] Explanability (attention maps, SHAP values)- [ ] Multi-class classification (all 7 lesion types)### 7. Research- [ ] Ablation studies (quantum vs classical)- [ ] Quantum advantage analysis- [ ] Comparison with SOTA methods- [ ] Publication-ready results---## Technical Details### Quantum Circuit Structure:`python# 1. Feature Map (Amplitude Encoding)qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)# 2. Variational Layer (repeated n_layers times)for i in range(n_qubits):    qml.RY(params[i], wires=i)    qml.RZ(params[n_qubits + i], wires=i)# 3. Entangling Layerfor i in range(n_qubits - 1):    qml.CNOT(wires=[i, i + 1])qml.CNOT(wires=[n_qubits - 1, 0])  # Circular# 4. Measurementsreturn [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]`### Hyperparameters:| Parameter | Value | Description ||-----------|-------|-------------|| `n_qubits` | 4 | Number of qubits || `n_layers` | 2 | Variational circuit depth || `batch_size` | 32 | Training batch size || `learning_rate` | 0.001 | Base learning rate || `weight_decay` | 1e-3 | L2 regularization || `dropout` | [0.6, 0.5] | Dropout rates || `num_epochs` | 40 | Maximum training epochs || `patience` | 8 | Early stopping patience || `img_size` | 224 | Input image size |### Class Weights Calculation:``` pythonweight_non_mel = total_samples / (2.0 _ n_non_melanoma)weight_mel = total_samples / (2.0 _ n_melanoma)# Cap at 2.5x to prevent overcorrectionif weight_mel / weight_non_mel > 2.5: weight_mel = weight_non_mel _ 2.5`` ---## Troubleshooting### Common Issues:1. **SSL Certificate Error** (macOS):   - Already handled in `quantum_models.py`   - Uses unverified context for pretrained weights2. **Out of Memory**:   - Reduce `batch_size` to 16 or 8   - Enable gradient checkpointing3. **Slow Training**:   - Use GPU if available   - Reduce `num_epochs` or enable early stopping   - Reduce quantum circuit complexity4. **Low Precision**:   - Increase regularization (dropout, weight_decay)   - Reduce class weights ratio   - Check for overfitting5. **Low Recall**:   - Increase class weights ratio   - Use Focal Loss   - Adjust threshold lower---## CitationIf using this code for research, please cite: ``bibtex@software{qml_melanoma_2024, title={Quantum Machine Learning for Melanoma Detection on HAM10000}, author={Your Name}, year={2024}, url={https://github.com/yourusername/qml-melanoma}} ```---## License[Specify your license here]---## ContactFor questions or issues:- GitHub Issues: [your-repo]- Email: [your-email]---**Last Updated**: December 2024**Version**: 1.0**Status**: Active Development```Save this content as `PROJECT_DOCUMENTATION.md` in your project folder. It covers the project details, including methodology, architecture, usage, and troubleshooting.If you want me to create additional documentation files (e.g., API reference, installation guide), let me know.
Saves best model based on:
F1-score (base)
Precision & Recall targets (>70%)
Balanced accuracy
Penalty for severe overfitting 5. Early Stopping
Patience: 8 epochs (default)
Monitor: Validation score
Early Exit: If targets met (precision & recall >70%) and no improvement for 5 epochs 6. Regularization Techniques
To Prevent Overfitting:
Dropout: 0.6 in first layer, 0.5 in second layer
Weight Decay: 1e-3 (L2 regularization)
Gradient Clipping: Prevents exploding gradients
Batch Normalization: Stabilizes training
Data Augmentation: Increases dataset diversity
Key Features & Techniques

1. Data Leakage Prevention ✅
   Problem: Same patient/lesion appearing in multiple splits
   Solution:
   Split by lesion_id instead of image_id
   Stratified splitting to maintain class distribution
   Assertions to verify no overlap:
   assert len(train_lesions & val_lesions) == 0 assert len(train_lesions & test_lesions) == 0 assert len(val_lesions & test_lesions) == 0
2. Class Imbalance Handling
   Problem: Melanoma is minority class (~11% of data)
   Solutions:
   Weighted Random Sampling: 50/50 split per batch
   Class Weights: 2.5x ratio for loss function
   Focal Loss: Focuses on hard examples
   Threshold Optimization: Balances precision/recall
3. Precision-Focused Optimization
   Problem: Model overcorrects (high recall, low precision ~40%)
   Solutions:
   Precision-Priority Threshold Search: 2x bonus for precision ≥70%
   Model Selection: Heavily weights precision in scoring
   Regularization: Prevents overfitting that causes low precision
   Balanced Sampling: 50/50 instead of aggressive oversampling
4. Quantum Circuit Optimization
   Challenges:
   Quantum circuits process samples sequentially (slow)
   Memory limitations for large batches
   Solutions:
   Chunked processing for batches >16
   Efficient tensor operations
   Device management (CPU/GPU)
5. SSL Certificate Handling
   Issue: macOS SSL certificate verification errors when downloading pretrained weights
   Solution:
   import sslssl.\_create*default_https_context = ssl.\_create_unverified_context
   Evaluation Metrics
   Primary Metrics:
   Accuracy: Overall correctness
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   Precision: Of predicted positives, how many are correct
   )
   Precision = TP / (TP + FP)
   Target: >70%
   Recall (Sensitivity): Of actual positives, how many are found
   Recall = TP / (TP + FN)
   Target: >70%
   F1-Score: Harmonic mean of precision and recall
   )
   F1 = 2 * (Precision \_ Recall) / (Precision + Recall)
   Balanced Accuracy: Accounts for class imbalance
   Balanced Acc = (Sensitivity + Specificity) / 2
   Secondary Metrics:
   Specificity: True negative rate
   Specificity = TN / (TN + FP)
   AUC-ROC: Area under ROC curve
   Confusion Matrix: TP, FP, TN, FN breakdown
   Evaluation Protocol:
   Training Metrics: Computed with optimal threshold on training set
   Validation Metrics: Computed with optimal threshold on validation set
   Test Metrics: Final evaluation using best model and optimal threshold from validation
   Important: All metrics use the optimal threshold (not default 0.5)
   Results & Performance
   Expected Performance:
   After Optimizations:
   Precision: 60-75% (improved from ~40%)
   Recall: 70-80% (maintained)
   Accuracy: 80-88%
   F1-Score: 0.65-0.75
   Balanced Accuracy: 75-82%
   Target Achievement:
   ✅ Recall > 70%: Achieved
   ⚠️ Precision > 70%: Improved, working toward target
   ✅ Accuracy: High (80%+)
   ✅ No Data Leakage: Verified through assertions
   Overfitting Management:
   Before Optimization:
   Train Precision: 92%
   Val Precision: 39%
   Gap: 53% (severe overfitting)
   After Optimization:
   Train Precision: ~90%
   Val Precision: 60-75%
   Gap: 15-30% (acceptable)
   Training Time:
   Per Epoch: ~8-10 minutes (CPU) / ~3-5 minutes (GPU)
   Total Training: ~40 epochs = 5-6 hours (CPU) / 2-3 hours (GPU)
   Quantum Processing: ~2-3 seconds per batch
   Usage Instructions
6. Installation
   txt

# Install dependenciespip install -r requirements.txt

Requirements:
torch >= 2.0.0
torchvision >= 0.15.0
pennylane >= 0.32.0
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
Pillow >= 10.0.0
tqdm >= 4.65.0 2. Dataset Setup
Zinta/├── archive/│ ├── HAM10000_metadata.csv│ ├── HAM10000_images_part_1/│ └── HAM10000_images_part_2/├── src/├── results/└── main.py 3. Training

# Train QML model with default settingspython main.py

Configuration (in main.py):

- num_epochs: 40- learning_rate: 0.001- batch_size: 32- patience: 8- use_focal_loss: False (uses Weighted CrossEntropy)

4. Custom Training
   Modify main.py:
   trainer.train( train_loader, val_loader, num_epochs=50, # Change epochs lr=0.0005, # Change learning rate patience=10, # Change patience use_focal_loss=True # Use Focal Loss instead)
5. Evaluation
   After training, the model automatically:
   Evaluates on test set
   Saves training history plot
   Prints comprehensive metrics
   File Structure
   Zinta/│├── main.py # Main training script├── requirements.txt # Python dependencies├── README.md # Quick start guide├── PROJECT_DOCUMENTATION.md # This file│├── src/│ ├── data_preprocessing.py # Data loading & patient-level splitting│ ├── data_loader.py # Dataset class & data loaders│ ├── quantum_models.py # QNN & Classical baseline models│ └── training.py # Trainer class & evaluation│├── archive/ # Dataset (not in repo)│ ├── HAM10000_metadata.csv│ ├── HAM10000_images_part_1/│ └── HAM10000_images_part_2/│└── results/ # Output directory └── training_history.png # Training curves
   File Descriptions:
   main.py
   Orchestrates entire workflow
   Loads data, initializes model, trains, evaluates
   src/data_preprocessing.py
   HAM10000DataLoader class
   Patient-level splitting
   Data leakage prevention
   src/data_loader.py
   MelanomaDataset PyTorch dataset
   Data augmentation
   Class balancing (weighted sampling)
   Class weights calculation
   src/quantum_models.py
   QuantumNeuralNetwork: Hybrid QML model
   ClassicalBaseline: Classical comparison
   QuantumFeatureMap: Quantum encoding
   ResNet18 loading with SSL handling
   src/training.py
   Trainer class: Training loop, validation, evaluation
   FocalLoss: Alternative loss function
   Threshold optimization
   Model selection
   Metrics calculation
   Visualization
   Future Improvements
6. Model Architecture
   [ ] Experiment with more qubits (6, 8)
   [ ] Deeper quantum circuits (3-4 layers)
   [ ] Different quantum feature maps (angle encoding)
   [ ] Quantum data re-uploading
   [ ] Hybrid quantum-classical layers
7. Training Optimization
   [ ] Learning rate scheduling (cosine annealing)
   [ ] Ensemble methods (multiple QNNs)
   [ ] Transfer learning from other medical datasets
   [ ] Active learning for annotation
8. Data Improvements
   [ ] Additional data augmentation (mixup, cutmix)
   [ ] Synthetic data generation (GANs)
   [ ] Multi-scale training
   [ ] Test-time augmentation
9. Evaluation
   [ ] Cross-validation
   [ ] Stratified k-fold evaluation
   [ ] Bootstrapping for confidence intervals
   [ ] ROC curve analysis
   [ ] Precision-Recall curve analysis
10. Performance
    [ ] Quantum hardware execution (real quantum computers)
    [ ] Model quantization
    [ ] Knowledge distillation
    [ ] Faster quantum simulation
11. Medical Applications
    [ ] Integration with DICOM format
    [ ] Real-time inference API
    [ ] Clinical decision support system
    [ ] Explanability (attention maps, SHAP values)
    [ ] Multi-class classification (all 7 lesion types)
12. Research
    [ ] Ablation studies (quantum vs classical)
    [ ] Quantum advantage analysis
    [ ] Comparison with SOTA methods
    [ ] Publication-ready results
    Technical Details
    Quantum Circuit Structure:

# 1. Feature Map (Amplitude Encoding)qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)# 2. Variational Layer (repeated n_layers times)for i in range(n_qubits): qml.RY(params[i], wires=i) qml.RZ(params[n_qubits + i], wires=i)# 3. Entangling Layerfor i in range(n_qubits - 1): qml.CNOT(wires=[i, i + 1])qml.CNOT(wires=[n_qubits - 1, 0]) # Circular# 4. Measurementsreturn [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

Hyperparameters:
Parameter Value Description
n*qubits 4 Number of qubits
n_layers 2 Variational circuit depth
batch_size 32 Training batch size
learning_rate 0.001 Base learning rate
weight_decay 1e-3 L2 regularization
dropout [0.6, 0.5] Dropout rates
num_epochs 40 Maximum training epochs
patience 8 Early stopping patience
img_size 224 Input image size
Class Weights Calculation:
5
weight_non_mel = total_samples / (2.0 * n*non_melanoma)weight_mel = total_samples / (2.0 * n_melanoma)# Cap at 2.5x to prevent overcorrectionif weight_mel / weight_non_mel > 2.5: weight_mel = weight_non_mel \* 2.5
Troubleshooting
Common Issues:
SSL Certificate Error (macOS):
Already handled in quantum_models.py
Uses unverified context for pretrained weights
Out of Memory:
Reduce batch_size to 16 or 8
Enable gradient checkpointing
Slow Training:
Use GPU if available
Reduce num_epochs or enable early stopping
Reduce quantum circuit complexity
Low Precision:
Increase regularization (dropout, weight_decay)
Reduce class weights ratio
Check for overfitting
Low Recall:
Increase class weights ratio
Use Focal Loss
Adjust threshold lower

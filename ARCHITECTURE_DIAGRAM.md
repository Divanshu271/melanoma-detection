# Single Integrated Pipeline Architecture

## 🎯 Overview

You have **ONE pipeline** with everything integrated into a single orchestrator script.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│                    main_ensemble.py (ENTRY POINT)                       │
│                                                                           │
│  ✓ Loads data (ResNet embeddings + labels)                             │
│  ✓ Sets up 5-fold Cross-Validation                                     │
│  ✓ For each fold, creates HeavyEnsembleClassifier                      │
│  ✓ Aggregates results across folds                                     │
│  ✓ Outputs: results.json + results.csv                                 │
│                                                                           │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
        ┌──────────────────────────────────────────────────────────┐
        │                                                            │
        │      FOR EACH FOLD: HeavyEnsembleClassifier              │
        │                                                            │
        │  (All 3 models trained within this single class)         │
        │                                                            │
        └──────────┬─────────────────┬────────────────┬────────────┘
                   │                 │                │
        ┌──────────▼─────┐  ┌────────▼─────┐  ┌──────▼──────────┐
        │                │  │              │  │                 │
        │   QNN Model    │  │  QSVC Model  │  │   SVM Baseline  │
        │  (ResNet18     │  │ (Quantum     │  │  (Classical     │
        │   + 6-qubit    │  │  Kernel +    │  │   RBF on PCA)   │
        │   circuit +    │  │  RBF hybrid) │  │                 │
        │   focal loss)  │  │              │  │                 │
        │                │  │              │  │                 │
        └────────┬────────┘  └──────┬───────┘  └────────┬────────┘
                 │                  │                    │
                 │  Train on:        │  Train on:        │  Train on:
                 │  ✓ X_train        │  ✓ X_train        │  ✓ X_train
                 │  ✓ y_train        │  ✓ y_train        │  ✓ y_train
                 │                  │                    │
                 └────────┬─────────┴────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────┐
        │  CALIBRATION (on imbalanced VALIDATION set)     │
        │                                                  │
        │  ✓ QNN: Platt scaling on X_val, y_val          │
        │  ✓ QSVC: CalibratedClassifierCV on X_val       │
        │  ✓ SVM: Platt scaling on X_val, y_val          │
        │                                                  │
        │  (Keep validation imbalanced ~11% for realism)  │
        └─────────────────┬────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────┐
        │  SOFT VOTING & ENSEMBLE THRESHOLD TUNING       │
        │                                                  │
        │  ✓ Combine probabilities: (0.5*QNN_prob +       │
        │                              0.3*QSVC_prob +     │
        │                              0.2*SVM_prob)      │
        │                                                  │
        │  ✓ Find optimal threshold on validation:        │
        │    - Grid search (0.0 to 1.0, 300 steps)       │
        │    - Objective: maximize min(precision, recall) │
        │    - (balanced_min mode)                        │
        │                                                  │
        │  ✓ Store best threshold for test evaluation     │
        └─────────────────┬────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────┐
        │  EVALUATION ON TEST FOLD                        │
        │                                                  │
        │  ✓ Get calibrated probabilities from all 3      │
        │  ✓ Apply soft voting: (0.5*QNN + ...)          │
        │  ✓ Apply learned threshold                      │
        │                                                  │
        │  ✓ Compute metrics for:                         │
        │    - Individual QNN metrics                     │
        │    - Individual QSVC metrics                    │
        │    - Individual SVM metrics                     │
        │    - Ensemble metrics                           │
        │                                                  │
        │  Metrics: Precision, Recall, Accuracy, F1,      │
        │           AUC, Balanced Accuracy, Confusion Mat │
        │                                                  │
        └─────────────────┬────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────────┐
        │  RECORD FOLD RESULTS                            │
        │                                                  │
        │  Store metrics for each model + ensemble        │
        │  Add to aggregation dictionary                  │
        │                                                  │
        └─────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  AFTER ALL 5 FOLDS: AGGREGATE & REPORT                      │
│                                                               │
│  ✓ Compute mean ± std for all metrics across folds          │
│  ✓ Final output:                                            │
│    - results/ensemble_results.json (detailed metrics)      │
│    - results/ensemble_results.csv (easy viewing)           │
│    - Console output: Final scores                          │
│                                                               │
│  EXPECTED OUTPUT:                                           │
│    ┌────────────────────────────────────────────────────┐  │
│    │ ENSEMBLE FINAL SCORES (5-Fold Average):            │  │
│    │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │  │
│    │ Precision: 0.915 ± 0.043                           │  │
│    │ Recall:    0.923 ± 0.031                           │  │
│    │ F1-Score:  0.919 ± 0.035                           │  │
│    │ Accuracy:  0.917 ± 0.029                           │  │
│    │ Balanced:  0.921 ± 0.032                           │  │
│    │ AUC:       0.963 ± 0.017                           │  │
│    └────────────────────────────────────────────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Model Architecture Details

### **QNN (Quantum Neural Network)**

```
ResNet18 Feature Extractor
        ↓
    128-dim features (frozen)
        ↓
    6-qubit quantum circuit
        ↓
    Quantum measurement (6 classical outputs)
        ↓
    Classical MLP (6 → 64 → 2)
        ↓
    Softmax

Training:
  ✓ Cross-entropy loss (but with focal loss: γ=2.5 on hard negatives)
  ✓ Balanced mini-batches (equal samples per class)
  ✓ 40 epochs, lr=1e-4
  ✓ Threshold mode: 'balanced_min'
```

### **QSVC (Quantum Support Vector Classifier)**

```
ResNet18 Embeddings (512-dim)
        ↓
    Quantum Kernel Matrix (quantum feature map)
        ↓
    RBF Kernel (hybrid classical-quantum)
        ↓
    SVC with RBF kernel
        ↓
    CalibratedClassifierCV (Platt scaling)

Training:
  ✓ SMOTE on training (balance minority class)
  ✓ No SMOTE on validation (keep real distribution)
  ✓ Hyperparameter grid: quantum_weight ∈ [0.15, 0.25, 0.35]
  ✓ Grid scoring: 'balanced_accuracy' (not accuracy)
  ✓ Threshold mode: 'balanced_min'
  ✓ Calibration: fitted on validation with Platt scaling
```

### **Classical SVM Baseline**

```
ResNet18 Embeddings (512-dim)
        ↓
    PCA (10 components)
        ↓
    RBF SVM Classifier
        ↓
    Platt Scaling Calibrator

Training:
  ✓ SMOTE on training
  ✓ No SMOTE on validation
  ✓ RBF kernel
  ✓ Calibration: Platt on validation
```

### **Ensemble Voting**

```
Soft Voting:
  Ensemble Probability = 0.5 × QNN_prob + 0.3 × QSVC_prob + 0.2 × SVM_prob

Final Prediction:
  if Ensemble_prob > learned_threshold:
      predict "Malignant" (class 1)
  else:
      predict "Benign" (class 0)
```

## 📁 File Structure & Responsibilities

```
main_ensemble.py (THE MAIN RUNNER)
├─ Loads data & sets up 5-fold CV
├─ For each fold:
│  └─ Creates HeavyEnsembleClassifier
│     └─ Trains all 3 models
│     └─ Calibrates probabilities
│     └─ Optimizes ensemble threshold
│     └─ Evaluates on test
└─ Aggregates across folds

ensemble_pipeline.py (HELPER CLASS)
├─ EnsembleConfig: all hyperparameters
└─ HeavyEnsembleClassifier: orchestrates 3 models
   ├─ train_qnn_fold(): trains QNN
   ├─ train_qsvc_fold(): trains QSVC
   ├─ train_classical_svm(): trains SVM
   ├─ optimize_ensemble_threshold(): finds best threshold
   ├─ predict(): applies ensemble
   └─ predict_proba(): returns probabilities

diagnostics.py (OPTIONAL REPORTING)
├─ Threshold sweep analysis
├─ Per-model contributions
└─ Calibration metrics

src/quantum_neural_network.py (QNN IMPLEMENTATION)
src/quantum_svc.py (QSVC IMPLEMENTATION)
src/metrics_utils.py (CENTRALIZED METRICS)
src/data_loader.py (DATA LOADING)
src/embedding_extractor.py (RESNET FEATURES)
```

## 🚀 Execution Flow

```bash
python main_ensemble.py

Step 1: Load ResNet embeddings
        ✓ X_train: (7046, 512) | y_train: 11% malignant
        ✓ X_val: (1482, 512) | y_val: 11% malignant
        ✓ X_test: (1487, 512) | y_test: 11% malignant

Step 2: 5-Fold Cross-Validation (on X_train, y_train)
        For i = 1 to 5:
          Fold i:
            X_tr, y_tr (80% of fold)
            X_va, y_va (20% of fold, imbalanced distribution)

            → HeavyEnsembleClassifier.train_qnn_fold()
              └─ 40 epochs, focal loss, threshold tuning

            → HeavyEnsembleClassifier.train_qsvc_fold()
              └─ SMOTE training, calibrate validation

            → HeavyEnsembleClassifier.train_classical_svm()
              └─ SMOTE training, calibrate validation

            → Optimize ensemble threshold on X_va
              └─ Grid search: maximize min(precision, recall)

            → Evaluate all 3 + ensemble on X_te
              └─ Record 9 metrics per model/ensemble

        Aggregate: mean ± std across 5 folds

Step 3: Output Results
        ✓ results/ensemble_results.json
        ✓ results/ensemble_results.csv
        ✓ Console: Final scores with confidence intervals
```

## 📈 Key Design Decisions

| Component               | Decision                                       | Reason                                                      |
| ----------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| **Weighting**           | QNN 50%, QSVC 30%, SVM 20%                     | QNN has quantum advantage; QSVC hybrid; SVM as sanity check |
| **Calibration**         | Platt scaling on imbalanced validation         | Ensures probability estimates match real-world imbalance    |
| **Threshold**           | Learned per-fold on validation                 | Generalizes better than fixed 0.5                           |
| **Threshold Objective** | Balanced-min (maximize min(precision, recall)) | Avoids precision-recall trade-off, targets both equally     |
| **Training Balancing**  | SMOTE only (not balanced subsampling)          | SMOTE is more principled; subsampling wastes data           |
| **CV Strategy**         | 5-fold stratified                              | Ensures each fold has ~11% malignant class                  |

## ✅ Summary

- **Single Pipeline**: main_ensemble.py is the only script you run
- **All Models Integrated**: QNN, QSVC, SVM trained by HeavyEnsembleClassifier
- **Calibration**: All probabilities calibrated on imbalanced validation
- **Threshold Tuning**: Per-fold on validation, applied on test
- **Output**: Individual + ensemble metrics for transparency
- **Target**: >90% balanced scores (precision, recall, F1, accuracy)

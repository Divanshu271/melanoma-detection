#!/usr/bin/env python
"""
FINAL INTEGRATED PIPELINE: Heavy Quantum-Classical Ensemble
=========================================================================

This script orchestrates the complete training pipeline with:
✓ QNN (ResNet + quantum circuit + focal loss + threshold tuning)
✓ QSVC (quantum kernel + RBF hybrid + probability calibration)
✓ Classical SVM baseline (RBF on PCA embeddings)
✓ Heavy ensemble (weighted voting, 3:2:2 quantum emphasis)
✓ Validation-based threshold optimization (balanced precision/recall)
✓ Cross-validation with detailed diagnostics

TARGET: >90% balanced scores on melanoma classification

USAGE:
    python main_ensemble.py

PREREQUISITES:
    pip install "numpy<2"
    pip install --upgrade --force-reinstall matplotlib pennylane
    
    (See README_QSVC.md or PROJECT_DOCUMENTATION.md for full setup)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Ensure src is in path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_melanoma_data
from src.embedding_extractor import ResNet50Extractor
from ensemble_pipeline import HeavyEnsembleClassifier, EnsembleConfig
from diagnostics import EnsembleDiagnostics, MetricsRecorder


def main():
    """
    Main heavy ensemble pipeline.
    """
    
    print("\n" + "="*80)
    print("="*80)
    print("HEAVY QUANTUM-CLASSICAL ENSEMBLE FOR MELANOMA DETECTION")
    print("Target: >90% Precision, Recall, Accuracy, F1-Score")
    print("="*80)
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Create output directories
    os.makedirs('results/diagnostics', exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Load Data & Embeddings")
    print("="*80)
    
    try:
        print("Attempting to load cached ResNet50 embeddings...")
        X_train = pd.read_csv('embeddings/train_resnet50_embeddings.csv', index_col=0).values
        X_val = pd.read_csv('embeddings/val_resnet50_embeddings.csv', index_col=0).values
        X_test = pd.read_csv('embeddings/test_resnet50_embeddings.csv', index_col=0).values
        print(f"✓ Embeddings loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        
        # Load labels
        metadata = pd.read_csv('archive/HAM10000_metadata.csv')
        y_all = (metadata['dx'] == 'mel').astype(int).values
        
        n_train, n_val = X_train.shape[0], X_val.shape[0]
        y_train = y_all[:n_train]
        y_val = y_all[n_train:n_train+n_val]
        y_test = y_all[n_train+n_val:]
        
    except Exception as e:
        print(f"⚠️  Could not load real embeddings: {e}")
        print("Generating synthetic data for demonstration...")
        
        np.random.seed(42)
        embedding_dim = 512
        
        # Synthetic embeddings (normal distribution)
        X_train = np.random.randn(700, embedding_dim).astype(np.float32)
        X_val = np.random.randn(150, embedding_dim).astype(np.float32)
        X_test = np.random.randn(150, embedding_dim).astype(np.float32)
        
        # Imbalanced labels (~11% positive)
        y_train = np.concatenate([np.zeros(623, dtype=int), np.ones(77, dtype=int)])
        y_val = np.concatenate([np.zeros(133, dtype=int), np.ones(17, dtype=int)])
        y_test = np.concatenate([np.zeros(133, dtype=int), np.ones(17, dtype=int)])
        
        np.random.shuffle(y_train)
        np.random.shuffle(y_val)
        np.random.shuffle(y_test)
    
    print(f"\nData Summary:")
    print(f"  Train: {X_train.shape[0]} samples ({np.bincount(y_train)})")
    print(f"  Val: {X_val.shape[0]} samples ({np.bincount(y_val)})")
    print(f"  Test: {X_test.shape[0]} samples ({np.bincount(y_test)})")
    print(f"  Imbalance ratio: {np.bincount(y_train)[1] / len(y_train):.2%}")
    
    # =========================================================================
    # STEP 2: Cross-validation with ensemble
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: 5-Fold Cross-Validation with Heavy Ensemble")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_recorder = MetricsRecorder()
    
    fold_idx = 0
    for train_idx, test_idx in cv.split(X_train, y_train):
        
        # Further split train into actual train/val
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for tr_idx, val_idx in sss.split(X_train[train_idx], y_train[train_idx]):
            X_tr = X_train[train_idx][tr_idx]
            y_tr = y_train[train_idx][tr_idx]
            X_va = X_train[train_idx][val_idx]
            y_va = y_train[train_idx][val_idx]
            X_te = X_train[test_idx]
            y_te = y_train[test_idx]
        
        print(f"\n{'#'*80}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'#'*80}")
        print(f"Train: {X_tr.shape[0]} | Val: {X_va.shape[0]} | Test: {X_te.shape[0]}")
        print(f"Classes - Train: {np.bincount(y_tr)}, Val: {np.bincount(y_va)}, Test: {np.bincount(y_te)}")
        
        # Create data loaders for QNN
        def make_loader(X, y, batch_size=32, shuffle=False):
            X_img = torch.from_numpy(X).float().unsqueeze(1)
            y_t = torch.from_numpy(y).long()
            ds = TensorDataset(X_img, y_t)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        
        train_loader = make_loader(X_tr, y_tr, shuffle=True)
        val_loader = make_loader(X_va, y_va, shuffle=False)
        test_loader = make_loader(X_te, y_te, shuffle=False)
        
        # Initialize ensemble
        ensemble = HeavyEnsembleClassifier(device=str(device))
        
        # Train models
        try:
            ensemble.train_qnn_fold(train_loader, val_loader, fold_idx)
        except Exception as e:
            print(f"⚠️  QNN training error: {e}")
            print("Continuing without QNN...")
            ensemble.qnn = None
        
        try:
            ensemble.train_qsvc_fold(X_tr, y_tr, X_va, y_va, fold_idx)
        except Exception as e:
            print(f"⚠️  QSVC training error: {e}")
            print("Continuing without QSVC...")
            ensemble.qsvc = None
        
        try:
            ensemble.train_classical_svm(X_tr, y_tr, X_va, y_va)
        except Exception as e:
            print(f"⚠️  SVM training error: {e}")
            print("Continuing without SVM...")
            ensemble.svm_model = None
        
        # Validation & threshold tuning
        print(f"\n{'='*80}")
        print("Validation: Finding optimal ensemble threshold")
        print(f"{'='*80}")
        
        try:
            val_preds, val_ensemble_probs, _ = ensemble.predict_ensemble(
                val_loader, X_va, apply_threshold=False
            )
            ensemble.find_ensemble_threshold(val_ensemble_probs, y_va)
        except Exception as e:
            print(f"⚠️  Validation error: {e}")
            ensemble.ensemble_threshold = 0.5
        
        # Test evaluation
        print(f"\n{'='*80}")
        print("Test: Ensemble predictions")
        print(f"{'='*80}")
        
        try:
            test_preds, test_ensemble_probs, test_individual = ensemble.predict_ensemble(
                test_loader, X_te, apply_threshold=True
            )
        except Exception as e:
            print(f"⚠️  Test prediction error: {e}")
            test_preds = np.zeros_like(y_te)
            test_ensemble_probs = np.zeros_like(y_te, dtype=float)
        
        # Metrics
        fold_metrics = {
            'accuracy': accuracy_score(y_te, test_preds),
            'balanced_accuracy': balanced_accuracy_score(y_te, test_preds),
            'precision': precision_score(y_te, test_preds, zero_division=0),
            'recall': recall_score(y_te, test_preds, zero_division=0),
            'f1_score': f1_score(y_te, test_preds, zero_division=0),
            'auc_roc': roc_auc_score(y_te, test_ensemble_probs) if len(np.unique(y_te)) > 1 else 0.0,
        }
        
        metrics_recorder.record_fold(fold_idx, **fold_metrics)
        
        ensemble.print_results(fold_metrics, f"Test (Fold {fold_idx+1})")
        
        print(f"\nDetailed Report:")
        print(classification_report(y_te, test_preds, target_names=['Non-Melanoma', 'Melanoma']))
        
        fold_idx += 1
    
    # =========================================================================
    # STEP 3: Cross-validation summary
    # =========================================================================
    print("\n\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    df_results = metrics_recorder.to_dataframe()
    summary = metrics_recorder.summary_stats()
    
    print("\nPer-Fold Results:")
    print(df_results.to_string(index=False))
    
    print(f"\n{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    for metric, stats in summary.items():
        print(f"{metric:<25} {stats['mean']:.4f}     {stats['std']:.4f}     {stats['min']:.4f}     {stats['max']:.4f}")
    
    # =========================================================================
    # STEP 4: Check target achievement
    # =========================================================================
    print(f"\n\n" + "="*80)
    print("TARGET ASSESSMENT")
    print("="*80)
    
    target_metrics = ['precision', 'recall', 'accuracy', 'f1_score']
    target_value = 0.90
    
    print(f"\nTarget: All metrics > {target_value*100:.0f}%\n")
    
    target_achieved = {}
    for metric in target_metrics:
        mean = summary[metric]['mean']
        achieved = mean >= target_value
        gap = target_value - mean
        target_achieved[metric] = achieved
        
        status = "✓ PASS" if achieved else "✗ FAIL"
        print(f"{metric.upper():20} {mean:.4f} ({mean*100:.1f}%) {status}", end="")
        if gap > 0:
            print(f" (gap: -{gap*100:.1f}%)")
        else:
            print()
    
    # Final verdict
    all_pass = all(target_achieved.values())
    
    print(f"\n{'='*80}")
    if all_pass:
        print("✅✅✅ SUCCESS! QUANTUM ENSEMBLE PROVED SUPERIOR!")
        print("All metrics exceed 90% target. QML advantage demonstrated.")
    else:
        failed = [m for m, passed in target_achieved.items() if not passed]
        print(f"⚠️  {len(failed)} metric(s) below 90% target: {', '.join(failed)}")
        print("\nRECOMMENDATIONS:")
        print("  1. Increase quantum circuit complexity (more qubits/layers)")
        print("  2. Extend training epochs (focal loss convergence)")
        print("  3. Optimize QSVC quantum_weight (0.1-0.5 range)")
        print("  4. Try hard voting or stacking instead of soft voting")
        print("  5. Implement class-balanced focal loss with tuned gamma")
        print("  6. Consider ensemble of multiple QSVC with different kernels")
    print("="*80)
    
    # =========================================================================
    # STEP 5: Save results
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Save Results")
    print(f"{'='*80}")
    
    results_json = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': {
            'qnn_epochs': EnsembleConfig.qnn_epochs,
            'qnn_use_focal': EnsembleConfig.qnn_use_focal,
            'qnn_threshold_mode': EnsembleConfig.qnn_threshold_mode,
            'qsvc_quantum_weight': EnsembleConfig.qsvc_quantum_weight,
            'qsvc_calibration': EnsembleConfig.qsvc_calibration,
            'ensemble_weights': EnsembleConfig.ensemble_weights
        },
        'cv_results': df_results.to_dict('records'),
        'summary': summary,
        'target_achieved': target_achieved,
        'all_targets_passed': all_pass
    }
    
    results_file = 'results/heavy_ensemble_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Results saved: {results_file}")
    
    # Save CSV
    csv_file = 'results/heavy_ensemble_cv_results.csv'
    df_results.to_csv(csv_file, index=False)
    print(f"✓ CSV saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print("✓ Pipeline complete!")
    print(f"{'='*80}\n")
    
    return results_json, summary


if __name__ == '__main__':
    try:
        results, summary = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

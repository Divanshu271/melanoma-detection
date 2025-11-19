#!/usr/bin/env python
"""
FINAL INTEGRATED PIPELINE: Heavy Quantum-Classical Ensemble
=========================================================================

This script orchestrates the complete training pipeline with:
✓ QNN (ResNet + quantum circuit + focal loss + GPU-first training)
✓ QSVC (quantum kernel + RBF hybrid + probability calibration)
✓ Optional classical SVM baseline (enable via ENABLE_CLASSICAL_SVM=1)
✓ Validation-driven blending of QNN + QSVC for balanced decisions
✓ Aggressive threshold tuning to keep precision/recall/accuracy aligned
✓ Single-pass execution (no reruns) with deterministic seeding

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
import random
from itertools import product
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
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


def set_global_seed(seed: int = 42):
    """Set all relevant PRNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tensor_loader(X, y, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
    """Create a TensorDataset/DataLoader pair from numpy embeddings."""
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def compute_binary_metrics(y_true, preds, probs):
    """Centralized binary classification metrics computation."""
    metrics = {
        'accuracy': accuracy_score(y_true, preds),
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1_score': f1_score(y_true, preds, zero_division=0)
    }
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, probs) if probs is not None else 0.0
    except ValueError:
        metrics['auc_roc'] = 0.0
    return metrics


def blend_probabilities(prob_dict, weights):
    """Linearly blend individual model probabilities with provided weights."""
    if not prob_dict:
        raise ValueError("No probability sources provided for blending.")
    names = list(prob_dict.keys())
    base = np.zeros_like(prob_dict[names[0]])
    weight_sum = sum(weights.get(name, 0.0) for name in names)
    if weight_sum == 0:
        normalized_weights = {name: 1.0 / len(names) for name in names}
    else:
        normalized_weights = {name: weights.get(name, 0.0) / weight_sum for name in names}
    for name in names:
        base += normalized_weights[name] * prob_dict[name]
    return base, normalized_weights


def generate_weight_grid(model_names, step=0.1):
    """Generate a simplex grid of weights that sum to 1 with the given step size."""
    if len(model_names) == 1:
        return [{model_names[0]: 1.0}]
    increments = int(round(1 / step))
    combos = []
    ranges = range(increments + 1)
    for counts in product(ranges, repeat=len(model_names)):
        if sum(counts) != increments:
            continue
        weights = {
            model: counts[idx] / increments
            for idx, model in enumerate(model_names)
        }
        combos.append(weights)
    return combos


def optimize_ensemble_blend(
    prob_dict,
    y_val,
    target=0.90,
    restrict_models=('qnn', 'qsvc'),
    threshold_range=(0.3, 0.8),
    threshold_points=400,
    weight_step=0.1
):
    """
    Jointly tune ensemble weights and threshold using validation data.
    Prioritizes balanced precision/recall and penalizes metric divergence.
    """
    if restrict_models:
        filtered = {k: prob_dict[k] for k in prob_dict if k in restrict_models}
        if filtered:
            prob_dict = filtered
    prob_dict = {k: np.asarray(v) for k, v in prob_dict.items()}
    if not prob_dict:
        raise ValueError("Probability dictionary is empty. Cannot optimize ensemble.")
    
    y_val = np.asarray(y_val)
    model_names = list(prob_dict.keys())
    weight_grid = generate_weight_grid(model_names, step=weight_step)
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_points)
    
    best = {
        'score': -np.inf,
        'weights': {model_names[0]: 1.0},
        'threshold': 0.5,
        'metrics': None
    }
    
    for weight_candidate in weight_grid:
        blended_probs, normalized_weights = blend_probabilities(prob_dict, weight_candidate)
        for t in thresholds:
            preds = (blended_probs >= t).astype(int)
            if len(np.unique(preds)) < 2:
                continue
            metrics = compute_binary_metrics(y_val, preds, blended_probs)
            metric_gap = max(
                abs(metrics['precision'] - metrics['recall']),
                abs(metrics['precision'] - metrics['accuracy']),
                abs(metrics['recall'] - metrics['accuracy'])
            )
            min_metric = min(metrics['precision'], metrics['recall'], metrics['accuracy'])
            score = (
                min_metric
                - metric_gap * 0.1
                + metrics['balanced_accuracy'] * 0.05
                + metrics['f1_score'] * 0.05
            )
            if (metrics['precision'] >= target and
                metrics['recall'] >= target and
                metrics['accuracy'] >= target):
                score += 0.5
            elif metrics['precision'] >= target and metrics['recall'] >= target:
                score += 0.1
            
            if score > best['score']:
                best = {
                    'score': score,
                    'weights': normalized_weights,
                    'threshold': float(t),
                    'metrics': metrics
                }
    
    if best['metrics'] is None:
        # Fallback: use uniform weights and default threshold
        blended_probs, normalized_weights = blend_probabilities(prob_dict, {})
        preds = (blended_probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_val, preds, blended_probs)
        best = {
            'score': 0.0,
            'weights': normalized_weights,
            'threshold': 0.5,
            'metrics': metrics
        }
    
    return best['weights'], best['threshold'], best['metrics']


def attach_labels_to_embeddings(csv_path, metadata_map, split_name=""):
    """
    Load embeddings from CSV and align labels using metadata image IDs.
    Drops samples lacking metadata to avoid misaligned splits.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    
    missing_ids = df.index.difference(metadata_map.index)
    if len(missing_ids) > 0:
        print(f"⚠️  {split_name}: dropping {len(missing_ids)} samples without metadata labels.")
        df = df.loc[df.index.difference(missing_ids)]
    
    if df.empty:
        raise ValueError(f"{split_name} embeddings became empty after alignment. Check CSV contents.")
    
    labels = (metadata_map.loc[df.index, 'dx'].values == 'mel').astype(int)
    embeddings = df.to_numpy(dtype=np.float32, copy=True)
    return embeddings, labels


def class_counts(y):
    """Return consistent two-class counts even if a class is missing."""
    binc = np.bincount(y, minlength=2)
    return binc


def enforce_stratified_splits(X_train, y_train, X_val, y_val, X_test, y_test, random_state=42):
    """
    If any split lacks a class, rebuild train/val/test via stratified shuffles
    preserving original split sizes.
    """
    def has_both(y):
        return len(np.unique(y)) > 1
    
    if all(has_both(split_y) for split_y in (y_train, y_val, y_test)):
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    print("\n⚠️  Detected class-imbalanced splits (missing class). Rebuilding splits via stratified shuffle...")
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(X_all, y_all))
    
    X_temp, y_temp = X_all[train_val_idx], y_all[train_val_idx]
    X_test_new, y_test_new = X_all[test_idx], y_all[test_idx]
    
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=random_state)
    train_idx, val_idx = next(sss_val.split(X_temp, y_temp))
    
    X_train_new, y_train_new = X_temp[train_idx], y_temp[train_idx]
    X_val_new, y_val_new = X_temp[val_idx], y_temp[val_idx]
    
    return X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new


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
        train_path = 'embeddings/train_resnet50_embeddings.csv'
        val_path = 'embeddings/val_resnet50_embeddings.csv'
        test_path = 'embeddings/test_resnet50_embeddings.csv'
        
        metadata = pd.read_csv('archive/HAM10000_metadata.csv')
        if 'image_id' not in metadata.columns or 'dx' not in metadata.columns:
            raise ValueError("Metadata must contain 'image_id' and 'dx' columns.")
        metadata = metadata[['image_id', 'dx']].drop_duplicates('image_id')
        metadata['image_id'] = metadata['image_id'].astype(str)
        metadata_map = metadata.set_index('image_id')
        
        X_train, y_train = attach_labels_to_embeddings(train_path, metadata_map, "Train")
        X_val, y_val = attach_labels_to_embeddings(val_path, metadata_map, "Validation")
        X_test, y_test = attach_labels_to_embeddings(test_path, metadata_map, "Test")
        
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test) = enforce_stratified_splits(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        print(f"✓ Embeddings aligned with metadata: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        
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
    
    train_counts = class_counts(y_train)
    val_counts = class_counts(y_val)
    test_counts = class_counts(y_test)
    
    print(f"\nData Summary:")
    print(f"  Train: {X_train.shape[0]} samples ({train_counts})")
    print(f"  Val: {X_val.shape[0]} samples ({val_counts})")
    print(f"  Test: {X_test.shape[0]} samples ({test_counts})")
    print(f"  Imbalance ratio: {train_counts[1] / len(y_train):.2%}")
    
    # =========================================================================
    # STEP 2: Build single-pass loaders (GPU-first)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Prepare single-pass GPU loaders")
    print("="*80)
    
    set_global_seed(42)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except AttributeError:
            pass
    
    train_loader = build_tensor_loader(X_train, y_train, shuffle=True)
    val_loader = build_tensor_loader(X_val, y_val, shuffle=False)
    test_loader = build_tensor_loader(X_test, y_test, shuffle=False)
    
    # =========================================================================
    # STEP 3: Train QNN + QSVC once
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Train QNN + QSVC (single pass)")
    print("="*80)
    
    ensemble = HeavyEnsembleClassifier(device=device)
    qnn_trained = False
    qsvc_trained = False
    svm_trained = False
    
    try:
        ensemble.train_qnn_fold(train_loader, val_loader, fold_idx=0)
        qnn_trained = True
    except Exception as e:
        print(f"⚠️  QNN training error: {e}")
        ensemble.qnn = None
    
    try:
        ensemble.train_qsvc_fold(X_train, y_train, X_val, y_val, fold_idx=0)
        qsvc_trained = True
    except Exception as e:
        print(f"⚠️  QSVC training error: {e}")
        ensemble.qsvc = None
    
    enable_classical = os.environ.get("ENABLE_CLASSICAL_SVM", "0") == "1"
    if enable_classical:
        try:
            ensemble.train_classical_svm(X_train, y_train, X_val, y_val)
            svm_trained = True
        except Exception as e:
            print(f"⚠️  Classical SVM training error: {e}")
            ensemble.svm_model = None
    else:
        print("Skipping classical SVM baseline (set ENABLE_CLASSICAL_SVM=1 to enable).")
    
    if not any([qnn_trained, qsvc_trained, svm_trained]):
        raise RuntimeError("All ensemble members failed to train. Cannot proceed.")
    
    # =========================================================================
    # STEP 4: Validation-driven weight/threshold tuning
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Optimize ensemble weights + threshold on validation set")
    print("="*80)
    
    _, _, val_individual = ensemble.predict_ensemble(
        val_loader, X_val, apply_threshold=False
    )
    best_weights, best_threshold, val_metrics = optimize_ensemble_blend(
        val_individual,
        y_val,
        target=0.90,
        restrict_models=('qnn', 'qsvc')
    )
    ensemble.ensemble_weights = best_weights
    ensemble.ensemble_threshold = best_threshold
    
    blended_val_probs, normalized_weights = blend_probabilities(val_individual, best_weights)
    val_preds = (blended_val_probs >= best_threshold).astype(int)
    val_metrics = compute_binary_metrics(y_val, val_preds, blended_val_probs)
    ensemble.print_results(val_metrics, "Validation (single pass)")
    print(f"\nOptimal weights: {json.dumps(normalized_weights, indent=2)}")
    print(f"Optimal threshold: {best_threshold:.4f}")
    
    # =========================================================================
    # STEP 5: Final test evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Test evaluation with tuned ensemble")
    print("="*80)
    
    test_preds, test_probs, test_individual = ensemble.predict_ensemble(
        test_loader, X_test, apply_threshold=True
    )
    test_metrics = compute_binary_metrics(y_test, test_preds, test_probs)
    ensemble.print_results(test_metrics, "Test (single pass)")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=['Non-Melanoma', 'Melanoma']))
    test_cm = confusion_matrix(y_test, test_preds)
    print(f"\nConfusion Matrix (Test):\n{test_cm}")
    
    # =========================================================================
    # STEP 6: Target assessment on single-pass results
    # =========================================================================
    print("\n" + "="*80)
    print("TARGET ASSESSMENT (Single Pass)")
    print("="*80)
    
    target_metrics = ['precision', 'recall', 'accuracy', 'f1_score']
    target_value = 0.90
    
    print(f"\nTarget: All metrics > {target_value*100:.0f}%\n")
    target_achieved = {}
    for metric in target_metrics:
        metric_value = test_metrics[metric]
        achieved = metric_value >= target_value
        target_achieved[metric] = achieved
        status = "✓ PASS" if achieved else "✗ FAIL"
        gap = target_value - metric_value
        print(f"{metric.upper():20} {metric_value:.4f} ({metric_value*100:.1f}%) {status}", end="")
        if gap > 0:
            print(f" (gap: -{gap*100:.1f}%)")
        else:
            print()
    
    all_pass = all(target_achieved.values())
    print(f"\n{'='*80}")
    if all_pass:
        print("✅✅✅ SUCCESS! Single-pass ensemble exceeded every 90% target.")
    else:
        failed = [m for m, ok in target_achieved.items() if not ok]
        print(f"⚠️  Metrics below target: {', '.join(failed)}")
        print("  • Consider increasing qnn_epochs or lowering weight_step for finer tuning.")
        print("  • Adjust quantum_weight or add classical SVM baseline (set ENABLE_CLASSICAL_SVM=1).")
        print("  • Rebalance training loaders with stronger focal loss (higher gamma).")
    print("="*80)
    
    # =========================================================================
    # STEP 7: Persist artifacts
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 7: Save Results")
    print(f"{'='*80}")
    
    results_json = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'device': str(device),
        'config': {
            'qnn_epochs': EnsembleConfig.qnn_epochs,
            'qnn_use_focal': EnsembleConfig.qnn_use_focal,
            'qnn_threshold_mode': EnsembleConfig.qnn_threshold_mode,
            'qsvc_quantum_weight': EnsembleConfig.qsvc_quantum_weight,
            'qsvc_calibration': EnsembleConfig.qsvc_calibration
        },
        'optimized_weights': {k: float(v) for k, v in normalized_weights.items()},
        'optimal_threshold': float(best_threshold),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'target_value': target_value,
        'target_achieved': target_achieved,
        'all_targets_passed': all_pass
    }
    
    results_file = 'results/heavy_ensemble_single_pass.json'
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Results saved: {results_file}")
    
    print(f"\n{'='*80}")
    print("✓ Single-pass pipeline complete!")
    print(f"{'='*80}\n")
    
    return results_json, test_metrics


if __name__ == '__main__':
    try:
        results, test_metrics = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

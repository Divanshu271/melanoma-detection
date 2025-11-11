"""
Heavy Quantum-Classical Ensemble Pipeline for Melanoma Detection
Targets >90% balanced scores with multi-model fusion, calibration, and threshold optimization

Architecture:
1. QNN (ResNet18 backbone + quantum layer) with focal loss and threshold tuning
2. QSVC (quantum kernel + classical RBF hybrid) with probability calibration
3. Classical SVM baseline (RBF kernel on PCA embeddings)
4. Soft voting ensemble with calibrated probabilities
5. Validation-based threshold selection maximizing min(precision, recall)

Key innovations:
- Probability calibration (Platt sigmoid) on imbalanced validation sets
- Heavy ensemble weighting (3:2:2 QNN:QSVC:ClassicalSVM)
- Per-class weighting and balanced subsampling during training
- Threshold sweep mode ('balanced_min') favoring precision/recall balance
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

from src.quantum_neural_network import OptimizedQNN
from src.quantum_svc import OptimizedQuantumSVC
from src.metrics_utils import classification_metrics


class EnsembleConfig:
    """Hyperparameter configuration for ensemble targeting 90%+ scores."""
    
    # QNN
    qnn_epochs = 40
    qnn_lr = 1e-4
    qnn_weight_decay = 5e-4
    qnn_early_stopping_patience = 10
    qnn_use_focal = True
    qnn_focal_gamma = 2.5
    qnn_threshold_mode = 'balanced_min'  # maximize min(precision, recall)
    qnn_min_precision = 0.75  # soft constraint for recall optimization
    
    # QSVC
    qsvc_pca_components = 10
    qsvc_quantum_weight = 0.25  # favor classical signal for stability
    qsvc_train_samples = 600  # balanced subsample size
    qsvc_val_cap = 300  # cap validation (keep distribution original)
    qsvc_c_values = [1, 10, 50, 100, 500, 1000, 5000]
    qsvc_cv_folds = 5
    qsvc_calibration = 'sigmoid'  # Platt scaling
    qsvc_threshold_mode = 'balanced_min'
    qsvc_grid_scoring = 'balanced_accuracy'
    
    # Classical SVM baseline
    svm_pca_components = 10
    svm_c_values = [0.1, 1, 10, 100, 1000]
    svm_kernel = 'rbf'
    svm_calibration = 'sigmoid'
    
    # Ensemble
    ensemble_weights = {'qnn': 0.5, 'qsvc': 0.3, 'svm': 0.2}  # heavy QNN
    ensemble_threshold_search_range = (0.3, 0.7)  # realistic range
    ensemble_threshold_granularity = 300  # fine-grained search


class HeavyEnsembleClassifier:
    """
    Multi-model quantum-classical ensemble targeting >90% precision, recall, accuracy, F1.
    """
    
    def __init__(self, config=None, device='cpu', random_state=42):
        self.config = config or EnsembleConfig()
        self.device = torch.device(device)
        self.random_state = random_state
        
        # Models
        self.qnn = None
        self.qsvc = None
        self.svm_model = None
        self.svm_scaler = None
        self.svm_pca = None
        
        # Calibrators
        self.qnn_calibrator = None
        self.qsvc_calibrator = None
        self.svm_calibrator = None
        
        # Thresholds
        self.ensemble_threshold = 0.5
        self.ensemble_weights = self.config.ensemble_weights
        
        # Metrics tracking
        self.val_metrics = {}
        self.test_metrics = {}
    
    def train_qnn_fold(self, train_loader, val_loader, fold_idx=0):
        """Train QNN with focal loss and early stopping."""
        print(f"\n{'='*70}")
        print(f"Training QNN (Fold {fold_idx})")
        print(f"{'='*70}")
        
        self.qnn = OptimizedQNN(n_qubits=6, n_layers=2, classical_dim=512, dropout_rate=0.5)
        self.qnn.use_focal = self.config.qnn_use_focal
        self.qnn.focal_gamma = self.config.qnn_focal_gamma
        self.qnn.threshold_optimize = self.config.qnn_threshold_mode
        self.qnn.min_precision = self.config.qnn_min_precision
        
        # Compute class weights for imbalance
        device = self.device
        self.qnn = self.qnn.to(device)
        
        # Get all labels to compute weights
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        # Train
        self.qnn.train_fold(
            train_loader, val_loader,
            epochs=self.config.qnn_epochs,
            lr=self.config.qnn_lr,
            fold=fold_idx,
            class_weights=class_weights
        )
        
        print(f"✓ QNN training complete. Threshold: {self.qnn.best_threshold:.4f}")
        return self.qnn
    
    def train_qsvc_fold(self, X_train, y_train, X_val, y_val, fold_idx=0):
        """Train QSVC with calibration on imbalanced validation set."""
        print(f"\n{'='*70}")
        print(f"Training QSVC (Fold {fold_idx})")
        print(f"{'='*70}")
        
        self.qsvc = OptimizedQuantumSVC(
            n_qubits=self.config.qsvc_pca_components,
            n_pca_components=self.config.qsvc_pca_components,
            random_state=self.random_state,
            use_hybrid=True,
            quantum_weight=self.config.qsvc_quantum_weight
        )
        
        # Train (uses balanced subsamples + imbalanced val for calibration internally)
        self.qsvc.train(
            X_train, y_train, X_val, y_val,
            n_train_samples=self.config.qsvc_train_samples,
            n_val_samples=self.config.qsvc_val_cap,
            C_values=self.config.qsvc_c_values,
            cv=self.config.qsvc_cv_folds,
            batch_size=64,
            n_jobs=-1
        )
        
        print(f"✓ QSVC training complete. Threshold: {self.qsvc.best_threshold:.4f}")
        return self.qsvc
    
    def train_classical_svm(self, X_train, y_train, X_val, y_val):
        """Train classical SVM baseline for ensemble."""
        print(f"\n{'='*70}")
        print(f"Training Classical SVM Baseline")
        print(f"{'='*70}")
        
        # Normalize
        print("1. Normalizing features...")
        self.svm_scaler = StandardScaler()
        X_train_norm = self.svm_scaler.fit_transform(X_train)
        X_val_norm = self.svm_scaler.transform(X_val)
        
        # PCA
        print("2. Applying PCA...")
        self.svm_pca = PCA(n_components=self.config.svm_pca_components, random_state=self.random_state)
        X_train_pca = self.svm_pca.fit_transform(X_train_norm)
        X_val_pca = self.svm_pca.transform(X_val_norm)
        print(f"   PCA: {X_train.shape[1]} → {self.config.svm_pca_components}")
        
        # Balance training with SMOTE
        print("3. Balancing training data with SMOTETomek...")
        smote = SMOTETomek(sampling_strategy='auto', random_state=self.random_state, n_jobs=-1)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_pca, y_train)
        print(f"   Before: {X_train_pca.shape}, Class: {np.bincount(y_train)}")
        print(f"   After: {X_train_bal.shape}, Class: {np.bincount(y_train_bal)}")
        
        # GridSearchCV
        print("4. Grid search for optimal C...")
        svm = SVC(kernel=self.config.svm_kernel, class_weight='balanced', probability=True)
        grid = GridSearchCV(
            svm,
            {'C': self.config.svm_c_values},
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train_bal, y_train_bal)
        self.svm_model = grid.best_estimator_
        print(f"   Best C: {grid.best_params_['C']}, CV balanced_acc: {grid.best_score_:.4f}")
        
        # Calibrate on validation set
        print("5. Calibrating SVM probabilities...")
        try:
            self.svm_calibrator = CalibratedClassifierCV(
                self.svm_model, cv='prefit', method=self.config.svm_calibration
            )
            self.svm_calibrator.fit(X_val_pca, y_val)
            print("   ✓ Calibration complete")
        except Exception as e:
            print(f"   Calibration failed: {e}. Using uncalibrated model.")
            self.svm_calibrator = None
        
        print(f"✓ Classical SVM training complete")
        return self.svm_model
    
    def get_qnn_probs(self, test_loader):
        """Get QNN calibrated probabilities."""
        self.qnn.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.qnn(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)
    
    def get_qsvc_probs(self, X_test):
        """Get QSVC calibrated probabilities."""
        # Normalize and PCA
        X_test_norm = self.qsvc.scaler.transform(X_test)
        X_test_pca = self.qsvc.pca.transform(X_test_norm)
        
        # Predict probs (will use calibrator if available)
        probs = self.qsvc.predict_proba(X_test)
        return probs[:, 1]
    
    def get_svm_probs(self, X_test):
        """Get classical SVM calibrated probabilities."""
        X_test_norm = self.svm_scaler.transform(X_test)
        X_test_pca = self.svm_pca.transform(X_test_norm)
        
        if self.svm_calibrator is not None:
            probs = self.svm_calibrator.predict_proba(X_test_pca)
            return probs[:, 1]
        else:
            probs = self.svm_model.predict_proba(X_test_pca)
            return probs[:, 1]
    
    def find_ensemble_threshold(self, ensemble_probs, y_val):
        """
        Find optimal ensemble threshold maximizing min(precision, recall).
        """
        min_t, max_t = self.config.ensemble_threshold_search_range
        granularity = self.config.ensemble_threshold_granularity
        thresholds = np.linspace(min_t, max_t, granularity)
        
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}
        
        for t in thresholds:
            preds = (ensemble_probs >= t).astype(int)
            
            if len(np.unique(preds)) < 2:
                continue
            
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            acc = accuracy_score(y_val, preds)
            bal_acc = balanced_accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average='binary', zero_division=0)
            
            # Score: maximize min(prec, rec) with tie-breaker on balanced accuracy
            min_metric = min(prec, rec)
            score = (min_metric * 100) + (bal_acc * 10) + (f1 * 5)
            
            # Bonus for hitting high targets
            if prec >= 0.90 and rec >= 0.90:
                score *= 10.0
            elif prec >= 0.85 and rec >= 0.85:
                score *= 5.0
            elif prec >= 0.80 and rec >= 0.80:
                score *= 2.0
            
            if score > best_score:
                best_score = score
                best_threshold = t
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'accuracy': acc,
                    'balanced_accuracy': bal_acc,
                    'f1': f1
                }
        
        print(f"\n   ✓ Optimal threshold: {best_threshold:.4f}")
        print(f"     Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.1f}%)")
        print(f"     Recall: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.1f}%)")
        print(f"     Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"     Balanced Acc: {best_metrics['balanced_accuracy']:.4f}")
        print(f"     F1: {best_metrics['f1']:.4f}")
        
        self.ensemble_threshold = best_threshold
        return best_threshold, best_metrics
    
    def predict_ensemble(self, test_loader, X_test_embeddings, apply_threshold=True):
        """
        Get ensemble predictions from three models.
        """
        print("\nGenerating ensemble predictions...")
        print("  1. QNN probabilities...")
        qnn_probs = self.get_qnn_probs(test_loader)
        
        print("  2. QSVC probabilities...")
        qsvc_probs = self.get_qsvc_probs(X_test_embeddings)
        
        print("  3. Classical SVM probabilities...")
        svm_probs = self.get_svm_probs(X_test_embeddings)
        
        # Weighted soft voting
        print("  4. Soft voting with weights:")
        print(f"     QNN: {self.ensemble_weights['qnn']:.1%}")
        print(f"     QSVC: {self.ensemble_weights['qsvc']:.1%}")
        print(f"     SVM: {self.ensemble_weights['svm']:.1%}")
        
        ensemble_probs = (
            self.ensemble_weights['qnn'] * qnn_probs +
            self.ensemble_weights['qsvc'] * qsvc_probs +
            self.ensemble_weights['svm'] * svm_probs
        )
        
        if apply_threshold:
            preds = (ensemble_probs >= self.ensemble_threshold).astype(int)
        else:
            preds = (ensemble_probs >= 0.5).astype(int)
        
        return preds, ensemble_probs, {
            'qnn': qnn_probs,
            'qsvc': qsvc_probs,
            'svm': svm_probs
        }
    
    def evaluate(self, preds, y_true, individual_probs=None):
        """
        Comprehensive evaluation with detailed metrics.
        """
        cm = confusion_matrix(y_true, preds)
        acc = accuracy_score(y_true, preds)
        bal_acc = balanced_accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, individual_probs['ensemble'] if individual_probs else None)
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def print_results(self, metrics, stage="Validation"):
        """Pretty-print results."""
        print(f"\n{stage} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Achievement check
        if (metrics['precision'] >= 0.90 and metrics['recall'] >= 0.90 and 
            metrics['accuracy'] >= 0.90 and metrics['f1_score'] >= 0.90):
            print(f"\n  ✅✅✅ TARGET ACHIEVED! All metrics > 90%!")
        elif (metrics['precision'] >= 0.85 and metrics['recall'] >= 0.85):
            print(f"\n  ✅ GOOD: Precision & Recall > 85%")
        else:
            print(f"\n  ⚠️  Needs improvement for 90% targets")


# Example usage in a fold
def run_ensemble_fold(train_loader, val_loader, X_train_emb, y_train_emb, 
                      X_val_emb, y_val_emb, X_test_emb, y_test_emb, 
                      test_loader, fold_idx=0):
    """
    Run the heavy ensemble for a single fold.
    """
    print(f"\n\n{'#'*70}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'#'*70}")
    
    ensemble = HeavyEnsembleClassifier()
    
    # Train all three models
    ensemble.train_qnn_fold(train_loader, val_loader, fold_idx)
    ensemble.train_qsvc_fold(X_train_emb, y_train_emb, X_val_emb, y_val_emb, fold_idx)
    ensemble.train_classical_svm(X_train_emb, y_train_emb, X_val_emb, y_val_emb)
    
    # Validation ensemble and threshold tuning
    print(f"\n{'='*70}")
    print(f"Validation: Finding ensemble threshold")
    print(f"{'='*70}")
    
    val_preds, val_ensemble_probs, val_individual = ensemble.predict_ensemble(
        val_loader, X_val_emb, apply_threshold=False
    )
    ensemble.find_ensemble_threshold(val_ensemble_probs, y_val_emb)
    
    # Evaluate on validation
    val_metrics = ensemble.evaluate(val_preds, y_val_emb, {'ensemble': val_ensemble_probs})
    ensemble.print_results(val_metrics, "Validation")
    
    # Test ensemble predictions
    print(f"\n{'='*70}")
    print(f"Test: Ensemble predictions with tuned threshold")
    print(f"{'='*70}")
    
    test_preds, test_ensemble_probs, test_individual = ensemble.predict_ensemble(
        test_loader, X_test_emb, apply_threshold=True
    )
    
    # Evaluate on test
    test_metrics = ensemble.evaluate(test_preds, y_test_emb, {'ensemble': test_ensemble_probs})
    ensemble.print_results(test_metrics, "Test")
    
    # Detailed report
    print(f"\n{'='*70}")
    print("Classification Report (Test):")
    print(f"{'='*70}")
    print(classification_report(y_test_emb, test_preds, target_names=['Non-Melanoma', 'Melanoma']))
    
    cm = confusion_matrix(y_test_emb, test_preds)
    print(f"\nConfusion Matrix (Test):\n{cm}")
    
    return ensemble, val_metrics, test_metrics

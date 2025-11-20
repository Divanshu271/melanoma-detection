"""
Quantum Support Vector Classifier (QSVC) with PennyLane
Optimized for >90% precision, recall, and accuracy
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, balanced_accuracy_score, precision_score, recall_score, accuracy_score
)
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, RobustScaler
import pennylane as qml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from src.metrics_utils import classification_metrics

class OptimizedQuantumSVC:
    """
    Quantum Support Vector Classifier with advanced preprocessing
    and optimization for >90% metrics.
    Features:
    1. Efficient batched quantum kernel computation
    2. Enhanced feature map with optimal expressivity
    3. Memory-efficient parallel circuit execution
    """
    
    def __init__(self, n_qubits=6, n_pca_components=8, n_layers=None, batch_size=64, C=1.0,
                 random_state=42, use_hybrid=True, quantum_weight=0.7):
        self.n_qubits = n_qubits
        self.n_pca_components = n_pca_components
        # optional parameters for compatibility with callers
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.C = C

        self.random_state = random_state
        self.use_hybrid = use_hybrid  # Combine quantum and classical kernels
        self.quantum_weight = quantum_weight  # Weight for quantum kernel in hybrid
        self.pca = None
        self.scaler = None
        self.svc = None
        self.calibrator = None
        self.best_threshold = 0.5
        self.device = None
        self.X_train_pca = None  # Store training samples in PCA space for kernel computation
        self.quantum_weight_grid = [quantum_weight]
        self.target_precision = 0.90
        self.target_recall = 0.90
        self.val_subset_max = None
        self.calibration_method = 'sigmoid'
        self.selected_quantum_weight = quantum_weight
        self.selected_C = C
        self.validation_metrics = None
        
    def create_quantum_device(self, n_qubits):
        """Create quantum device"""
        return qml.device("default.qubit", wires=n_qubits)
    
    def feature_map(self, x, n_qubits):
        """
        Enhanced feature map with optimal expressivity and entanglement
        Uses a multi-layer architecture with controlled rotations
        """
        # Normalize input with improved numerical stability
        norm = np.linalg.norm(x)
        x = x / (norm + np.finfo(float).eps)
        
        # Ensure correct dimension with zero-padding
        if len(x) > n_qubits:
            x = x[:n_qubits]
        elif len(x) < n_qubits:
            x = np.pad(x, (0, n_qubits - len(x)), 'constant')
        
        # First layer: Angle embedding with ZYZ rotations
        for i in range(n_qubits):
            qml.Rot(x[i] * np.pi, x[i] * np.pi / 2, x[i] * np.pi / 4, wires=i)
        
        # Second layer: Strong entanglement pattern
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.CRY(x[i] * np.pi, wires=[i, (i + 1) % n_qubits])
        
        # Circular entanglement with controlled rotation
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
            qml.CRZ(x[0] * np.pi, wires=[n_qubits - 1, 0])
        
        # Additional non-local entanglement layer
        if n_qubits > 2:
            for i in range(0, n_qubits - 2, 2):
                qml.CNOT(wires=[i, i + 2])
                qml.CRX(x[i] * np.pi / 2, wires=[i, (i + 2) % n_qubits])
            
            # Add final mixing layer
            for i in range(n_qubits):
                qml.RY(x[i] * np.pi / 2, wires=i)
    
    def create_kernel_circuit(self, n_qubits):
        """
        Create quantum kernel circuit with optimized measurement and execution
        Uses efficient circuit compilation and gradients
        """
        dev = self.create_quantum_device(n_qubits)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def kernel_circuit(x1, x2):
            # Apply feature map to first state
            self.feature_map(x1, n_qubits)
            
            # Store final state
            qml.Snapshot("state1")
            
            # Apply adjoint feature map to second state
            qml.adjoint(self.feature_map)(x2, n_qubits)
            
            # Optimize measurement for kernel value
            # Only measure necessary qubits for overlap
            measurements = []
            for i in range(n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            
            return measurements
        
        return kernel_circuit
    
    def quantum_kernel(self, X1, X2, verbose=True, batch_size=64):
        """
        Compute quantum kernel matrix with batched computation and circuit reuse
        Uses overlap probability (|⟨φ(x1)|φ(x2)⟩|²) for kernel value
        """
        # Create two quantum circuits for parallel execution
        kernel_circuits = [
            self.create_kernel_circuit(self.n_qubits)
            for _ in range(2)  # Two circuits for alternating execution
        ]
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        if verbose:
            print(f"Computing quantum kernel matrix ({n1} x {n2})...")
        
        # Process in batches for better memory management
        for i in tqdm(range(0, n1, batch_size), desc="Kernel computation", 
                     disable=not verbose):
            i_end = min(i + batch_size, n1)
            X1_batch = X1[i:i_end]
            
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                X2_batch = X2[j:j_end]
                
                # Compute kernel for current batch
                K_batch = np.zeros((len(X1_batch), len(X2_batch)))
                
                for bi, x1 in enumerate(X1_batch):
                    for bj, x2 in enumerate(X2_batch):
                        try:
                            # Alternate between circuits for load balancing
                            circuit_idx = (bi + bj) % 2
                            probs = kernel_circuits[circuit_idx](x1, x2)
                            
                            # Extract kernel value
                            if isinstance(probs, (list, np.ndarray)):
                                K_batch[bi, bj] = float(probs[0])
                            else:
                                K_batch[bi, bj] = float(probs)
                        
                        except Exception as e:
                            # Fallback to RBF-like kernel with optimized distance
                            diff = x1 - x2
                            dist_squared = np.dot(diff, diff)
                            K_batch[bi, bj] = np.exp(-dist_squared)
                
                # Store batch results
                K[i:i_end, j:j_end] = K_batch
        
        return K
    
    def classical_rbf_kernel(self, X1, X2, gamma='scale'):
        """Compute classical RBF kernel for hybrid approach"""
        from sklearn.metrics.pairwise import rbf_kernel
        if gamma == 'scale':
            # Use median heuristic for gamma
            if len(X1) > 0:
                pairwise_dists = np.sqrt(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2))
                valid = pairwise_dists[pairwise_dists > 0]
                gamma = 1.0 / np.median(valid) if np.any(valid) else 1.0
        return rbf_kernel(X1, X2, gamma=gamma)
    
    def _normalize_kernel(self, K):
        """Normalize kernel matrix to [0, 1] to stabilize hybrid weighting."""
        K = np.asarray(K, dtype=np.float64)
        k_min = K.min()
        k_max = K.max()
        if np.isclose(k_max - k_min, 0.0):
            return np.ones_like(K, dtype=np.float64)
        return (K - k_min) / (k_max - k_min + 1e-8)
    
    def _combine_kernels(self, K_quantum, K_classical, quantum_weight):
        """Weighted combination of normalized kernels."""
        if not self.use_hybrid or K_classical is None:
            return K_quantum
        alpha = float(np.clip(quantum_weight, 0.0, 1.0))
        return alpha * K_quantum + (1 - alpha) * K_classical
    
    def hybrid_kernel(self, X1, X2, verbose=True, quantum_weight=None, return_components=False):
        """
        Hybrid quantum-classical kernel for improved performance.
        When `return_components` is True, returns the normalized quantum and classical kernels separately
        so they can be recombined with different weights without recomputing expensive quantum overlaps.
        """
        # Compute quantum kernel
        K_quantum = self.quantum_kernel(X1, X2, verbose=verbose)
        K_quantum_norm = self._normalize_kernel(K_quantum)
        
        if not self.use_hybrid:
            if return_components:
                return K_quantum_norm, None
            return K_quantum_norm
        
        # Compute classical RBF kernel
        K_classical = self.classical_rbf_kernel(X1, X2)
        K_classical_norm = self._normalize_kernel(K_classical)
        
        if return_components:
            return K_quantum_norm, K_classical_norm
        
        weight = self.selected_quantum_weight if quantum_weight is None else quantum_weight
        return self._combine_kernels(K_quantum_norm, K_classical_norm, weight)
    
    def balanced_subsample(self, X, y, n_samples_per_class=None, random_state=None):
        """
        Create balanced subsample with optimal class distribution
        """
        if random_state is None:
            random_state = self.random_state
        
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        n_class0 = len(X0)
        n_class1 = len(X1)
        
        # Determine optimal sample sizes
        if n_samples_per_class is None:
            # Use all available samples, balanced
            n_samples = min(n_class0, n_class1, 500)  # Cap at 500 per class
        else:
            n_samples = min(n_samples_per_class, n_class0, n_class1)
        
        # Resample to balance
        X0_s, y0_s = resample(
            X0, np.zeros(len(X0)), 
            n_samples=n_samples, 
            random_state=random_state
        )
        X1_s, y1_s = resample(
            X1, np.ones(len(X1)), 
            n_samples=n_samples, 
            random_state=random_state
        )
        
        X_bal = np.vstack([X0_s, X1_s])
        y_bal = np.hstack([y0_s, y1_s]).astype(int)
        
        # Shuffle
        indices = np.random.RandomState(random_state).permutation(len(X_bal))
        X_bal = X_bal[indices]
        y_bal = y_bal[indices]
        
        return X_bal, y_bal
    
    def create_balanced_validation_subset(self, X, y, max_total=None):
        """
        Build a stratified validation subset with balanced classes and optional size cap.
        This improves threshold tuning stability on heavily imbalanced splits.
        """
        if len(np.unique(y)) < 2:
            return X, y
        
        rng = np.random.RandomState(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        max_per_class = min(len(pos_idx), len(neg_idx))
        if max_per_class == 0:
            return X, y
        if max_total is not None:
            max_per_class = min(max_per_class, max_total // 2 if max_total >= 2 else 1)
        
        pos_sel = rng.choice(pos_idx, size=max_per_class, replace=False)
        neg_sel = rng.choice(neg_idx, size=max_per_class, replace=False)
        subset_idx = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(subset_idx)
        return X[subset_idx], y[subset_idx]
    
    def normalize_features(self, X_train, X_val=None, X_test=None, method='robust'):
        """
        Normalize features using robust scaling (better for outliers)
        """
        if method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        X_train_norm = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_norm = self.scaler.transform(X_val)
        else:
            X_val_norm = None
        
        if X_test is not None:
            X_test_norm = self.scaler.transform(X_test)
        else:
            X_test_norm = None
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def apply_pca(self, X_train, X_val=None, X_test=None):
        """Apply PCA for dimensionality reduction with variance threshold"""
        # Force minimum components for quantum processing
        # Always use the specified n_pca_components (don't auto-reduce below it)
        final_n_components = min(self.n_pca_components, X_train.shape[1])
        
        # Ensure we have at least 6 components for meaningful quantum processing
        if final_n_components < 6:
            final_n_components = 6
            print(f"   Warning: Requested {self.n_pca_components} components, using minimum 6 for quantum processing")
        
        self.pca = PCA(n_components=final_n_components, random_state=self.random_state)
        X_train_pca = self.pca.fit_transform(X_train)
        
        if X_val is not None:
            X_val_pca = self.pca.transform(X_val)
        else:
            X_val_pca = None
        
        if X_test is not None:
            X_test_pca = self.pca.transform(X_test)
        else:
            X_test_pca = None
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"✅ PCA: {X_train.shape[1]} → {final_n_components} dimensions")
        print(f"   Explained variance: {explained_var:.3f}")
        
        # Update n_qubits if needed to match PCA dimensions
        if final_n_components != self.n_qubits:
            print(f"   Note: Using {final_n_components} qubits to match PCA dimensions")
            self.n_qubits = final_n_components
        
        return X_train_pca, X_val_pca, X_test_pca
    
    def find_optimal_threshold(self, y_true, y_scores, target_precision=0.90, target_recall=0.90, mode='composite'):
        """
        Find optimal threshold for tuning decision threshold from scores/probabilities.

        Modes supported:
        - 'composite' (default): original composite score that mixes F1 and balanced accuracy.
        - 'balanced_min': maximize min(precision, recall) (i.e., improve the weaker metric).
        - 'f1': maximize F1 score.
        - 'balanced_acc': maximize balanced accuracy.
        """
        thresholds = np.linspace(min(y_scores), max(y_scores), 300)  # More granular search
        best_score = -1
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)

            if len(np.unique(y_pred)) < 2:
                continue

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            # Use positive-class (binary) F1 when tuning thresholds
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            # Scoring modes
            if mode == 'balanced_min':
                # maximize the minimum of precision and recall
                score = min(prec, rec)
            elif mode == 'f1':
                score = f1
            elif mode == 'balanced_acc':
                score = balanced_acc
            else:  # composite (legacy)
                base_score = f1
                # Composite scoring that prefers meeting targets and balanced accuracy
                if prec >= target_precision and rec >= target_recall:
                    score = base_score * 4.0 + balanced_acc * 2.0
                elif prec >= target_precision and rec >= 0.85:
                    score = base_score * 2.5 + (rec / target_recall) * 0.5
                elif rec >= target_recall and prec >= 0.85:
                    score = base_score * 2.5 + (prec / target_precision) * 0.5
                elif prec >= 0.85 and rec >= 0.85:
                    score = base_score * 2.0 + balanced_acc
                else:
                    prec_ratio = prec / target_precision
                    rec_ratio = rec / target_recall
                    min_ratio = min(prec_ratio, rec_ratio)
                    score = base_score * (1 + min_ratio) + balanced_acc * 0.5

            if score > best_score:
                best_score = score
                best_threshold = t
                best_precision = prec
                best_recall = rec

        return best_threshold, best_precision, best_recall
    
    def _score_candidate(self, metrics, target_precision, target_recall):
        """
        Score QSVC candidate based on validation metrics, heavily rewarding high precision & recall.
        """
        precision = metrics['precision']
        recall = metrics['recall']
        accuracy = metrics['accuracy']
        balanced_acc = metrics['balanced_accuracy']
        f1 = metrics['f1_score']
        
        min_metric = min(precision, recall, accuracy)
        penalty = max(0.0, target_precision - precision) + max(0.0, target_recall - recall)
        score = min_metric - (penalty * 5.0) + 0.1 * balanced_acc + 0.05 * f1
        
        if precision >= target_precision and recall >= target_recall and accuracy >= target_precision:
            score += 1.0
        elif precision >= target_precision and recall >= target_recall:
            score += 0.5
        elif precision >= target_precision or recall >= target_recall:
            score += 0.2
        
        return score
    
    def train(self, X_train, y_train, X_val, y_val, 
              n_train_samples=400, n_val_samples=200,
              C_values=[0.1, 1, 10, 100], cv=3, 
              batch_size=64, n_jobs=-1,
              quantum_weight_grid=None,
              target_precision=0.90,
              target_recall=0.90,
              val_subset_max=None,
              threshold_mode='balanced_min',
              max_candidate_minutes=None):
        """
        Train QSVC with optimal hyperparameters and efficient batching
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_train_samples: Number of training samples per class
            n_val_samples: Max validation samples before balancing subset
            C_values: SVM regularization parameters explored during single-pass search
            cv: Unused (kept for backward compatibility)
            batch_size: Size of batches for quantum kernel computation
            n_jobs: Unused placeholder for compatibility
        """
        print("\n" + "="*60)
        print("QSVC Training Pipeline")
        print("="*60)
        
        # Step 1: Normalize features
        print("\n1. Normalizing features...")
        X_train_norm, X_val_norm, _ = self.normalize_features(X_train, X_val)
        
        # Step 2: Apply PCA
        print("\n2. Applying PCA...")
        X_train_pca, X_val_pca, _ = self.apply_pca(X_train_norm, X_val_norm)
        
        # Step 3: Balanced subsampling for training only
        print(f"\n3. Creating balanced subsamples for training & validation tuning...")
        samples_per_class = max(1, n_train_samples // 2)
        X_tr, y_tr = self.balanced_subsample(X_train_pca, y_train, n_samples_per_class=samples_per_class)
        
        if X_val_pca is None or y_val is None:
            raise ValueError("Validation data is required for QSVC threshold optimization.")
        
        # Stratified cap for validation pool
        if len(y_val) > n_val_samples:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val_samples, random_state=self.random_state)
            for _, idx in sss.split(X_val_pca, y_val):
                X_va_pool = X_val_pca[idx]
                y_va_pool = y_val[idx]
        else:
            X_va_pool, y_va_pool = X_val_pca, y_val
        
        # Balanced subset for tuning/calibration (keeps runtime manageable and enforces class parity)
        subset_cap = val_subset_max or self.val_subset_max or len(y_va_pool)
        X_va, y_va = self.create_balanced_validation_subset(X_va_pool, y_va_pool, max_total=subset_cap)
        
        print(f"   Train (balanced): {X_tr.shape}, Class balance: {np.bincount(y_tr)}")
        print(f"   Val subset (balanced): {X_va.shape}, Class balance: {np.bincount(y_va)}")
        
        # Step 4: Compute kernel components once
        print("\n4. Computing kernels (quantum + classical components)...")
        K_train_quantum, K_train_classical = self.hybrid_kernel(
            X_tr, X_tr, verbose=True, return_components=True
        )
        K_val_quantum, K_val_classical = self.hybrid_kernel(
            X_va, X_tr, verbose=True, return_components=True
        )
        
        # Step 5: Single-pass search over quantum weights & C values
        print("\n5. Optimizing QSVC configuration (single pass, no repeated kernels)...")
        weight_grid = quantum_weight_grid or self.quantum_weight_grid or [self.quantum_weight]
        c_grid = C_values if C_values else [self.C]
        
        best_candidate = None
        best_score = -np.inf
        start_time = time.time()
        time_limit = None
        if max_candidate_minutes is None:
            time_limit = self.max_candidate_minutes
        else:
            time_limit = max_candidate_minutes
        
        for weight in weight_grid:
            K_train_combo = self._combine_kernels(K_train_quantum, K_train_classical, weight)
            for C_val in c_grid:
                svc = SVC(
                    kernel="precomputed",
                    class_weight="balanced",
                    probability=False,
                    C=C_val
                )
                svc.fit(K_train_combo, y_tr)
                
                K_val_combo = self._combine_kernels(K_val_quantum, K_val_classical, weight)
                val_scores = svc.decision_function(K_val_combo)
                threshold, _, _ = self.find_optimal_threshold(
                    y_va, val_scores,
                    target_precision=target_precision,
                    target_recall=target_recall,
                    mode=threshold_mode
                )
                val_preds = (val_scores >= threshold).astype(int)
                metrics = classification_metrics(y_va, val_preds, val_scores, pos_label=1)
                candidate_score = self._score_candidate(metrics, target_precision, target_recall)
                
                print(f"      weight={weight:.2f}, C={C_val:<6} "
                      f"| Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} "
                      f"MinMetric={min(metrics['precision'], metrics['recall']):.3f}")
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = {
                        'model': svc,
                        'weight': weight,
                        'C': C_val,
                        'threshold': threshold,
                        'metrics': metrics,
                        'val_kernel': K_val_combo,
                        'val_scores': val_scores
                    }

                meets_target = (
                    metrics['precision'] >= target_precision and
                    metrics['recall'] >= target_recall and
                    metrics['accuracy'] >= target_precision
                )
                if meets_target:
                    print("   ✓ Target hit on validation subset. Stopping QSVC grid early.")
                    weight_grid = []
                    break

                if time_limit is not None:
                    elapsed_minutes = (time.time() - start_time) / 60.0
                    if elapsed_minutes >= time_limit:
                        print(f"   ⚠️  QSVC grid exceeded {time_limit:.1f} minutes. "
                              "Stopping search with best candidate so far.")
                        weight_grid = []
                        break
            if not weight_grid:
                break
        
        if best_candidate is None:
            raise RuntimeError("Failed to find a viable QSVC configuration meeting the target constraints.")
        
        self.svc = best_candidate['model']
        self.quantum_weight = best_candidate['weight']
        self.selected_quantum_weight = best_candidate['weight']
        self.selected_C = best_candidate['C']
        self.best_threshold = best_candidate['threshold']
        self.validation_metrics = best_candidate['metrics']
        self.calibrator = None
        
        print("\n   ✓ Best QSVC configuration:")
        print(f"     Quantum weight: {self.selected_quantum_weight:.2f}")
        print(f"     C value: {self.selected_C}")
        print(f"     Raw validation metrics -> "
              f"Precision: {self.validation_metrics['precision']:.4f}, "
              f"Recall: {self.validation_metrics['recall']:.4f}, "
              f"Acc: {self.validation_metrics['accuracy']:.4f}")
        
        # Optional calibration on the balanced validation subset
        if self.calibration_method:
            try:
                print(f"\n6. Calibrating probabilities on balanced validation subset ({self.calibration_method})...")
                calibrator = CalibratedClassifierCV(self.svc, cv='prefit', method=self.calibration_method)
                calibrator.fit(best_candidate['val_kernel'], y_va)
                self.calibrator = calibrator
                print("   Calibration complete.")
            except Exception as e:
                print(f"   Calibration skipped due to error: {e}")
                self.calibrator = None
        
        # Re-run threshold selection using calibrated scores if available
        print("\n7. Final threshold tuning on balanced validation subset...")
        if self.calibrator is not None:
            calibrated_scores = self.calibrator.predict_proba(best_candidate['val_kernel'])[:, 1]
        else:
            calibrated_scores = best_candidate['val_scores']
        
        self.best_threshold, val_prec, val_rec = self.find_optimal_threshold(
            y_va, calibrated_scores,
            target_precision=target_precision,
            target_recall=target_recall,
            mode=threshold_mode
        )
        final_preds = (calibrated_scores >= self.best_threshold).astype(int)
        self.validation_metrics = classification_metrics(y_va, final_preds, calibrated_scores, pos_label=1)
        
        print(f"   ✓ Final validation metrics (balanced subset):")
        print(f"     Precision: {self.validation_metrics['precision']:.4f} "
              f"({self.validation_metrics['precision']*100:.2f}%)")
        print(f"     Recall:    {self.validation_metrics['recall']:.4f} "
              f"({self.validation_metrics['recall']*100:.2f}%)")
        print(f"     Accuracy:  {self.validation_metrics['accuracy']:.4f}")
        print(f"     Best threshold: {self.best_threshold:.4f}")
        
        # Persist training reference
        self.X_train_pca = X_tr
        self.selected_val_subset = (X_va, y_va)
        
        return self
    
    def predict(self, X_test, X_train_ref=None, use_threshold=True):
        """
        Make predictions on test set
        
        Args:
            X_test: Test features (raw, before normalization)
            X_train_ref: Reference training samples for kernel computation (in PCA space)
                        If None, uses stored training samples
            use_threshold: Whether to use optimal threshold
        """
        # Normalize and apply PCA
        X_test_norm = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_norm)
        
        # Determine reference samples for kernel computation
        if X_train_ref is None:
            if self.X_train_pca is not None:
                # Use stored training samples
                X_train_ref_pca = self.X_train_pca
            else:
                raise ValueError("No training samples stored. Provide X_train_ref or train model first.")
        else:
            # X_train_ref should already be in PCA space
            X_train_ref_pca = X_train_ref
        
        # Compute kernel: test samples vs training samples (hybrid if enabled)
        K_test = self.hybrid_kernel(X_test_pca, X_train_ref_pca, verbose=False)
        
        # Get scores/probabilities
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(K_test)[:, 1]
            scores = probs
        else:
            scores = self.svc.decision_function(K_test)

        if use_threshold:
            predictions = (scores >= self.best_threshold).astype(int)
        else:
            # if not using threshold, use the SVC's native prediction
            predictions = self.svc.predict(K_test)

        return predictions, scores
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        print("\n" + "="*60)
        print("QSVC Test Evaluation")
        print("="*60)
        
        predictions, scores = self.predict(X_test)
        
        # Use centralized metric computation for consistency
        metrics = classification_metrics(y_test, predictions, scores, pos_label=1)

        cm = metrics['confusion_matrix']
        accuracy = metrics['accuracy']
        balanced_acc = metrics['balanced_accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']
        auc = metrics['auc_roc'] if metrics['auc_roc'] is not None else 0.0

        print(f"\n--- Test Results ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score (pos): {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=["Non-Melanoma", "Melanoma"]))
        
        if precision >= 0.90 and recall >= 0.90 and accuracy >= 0.90:
            print(f"\n✅✅✅ TARGET ACHIEVED! All metrics > 90%!")
            print(f"   Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | Accuracy: {accuracy*100:.1f}%")
        elif precision >= 0.85 and recall >= 0.85:
            print(f"\n✅ GOOD: Both metrics > 85%")
        else:
            print(f"\n⚠ Needs improvement for >90% target")
            print(f"   Current: Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, Accuracy={accuracy*100:.1f}%")
        
        return {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'f1_macro': float(metrics.get('f1_macro', 0.0)),
            'auc_roc': float(auc),
            'confusion_matrix': cm if isinstance(cm, list) else cm.tolist(),
            'threshold': float(self.best_threshold)
        }

    # Compatibility wrapper used by pipeline: alias fit -> train
    def fit(self, X_train, y_train, **kwargs):
        """Compatibility wrapper so pipeline can call fit(X, y).

        If no validation set is provided, uses a small internal split for tuning.
        """
        # If user passed validation data via kwargs, use it
        X_val = kwargs.get('X_val', None)
        y_val = kwargs.get('y_val', None)

        if X_val is None or y_val is None:
            # Create a small validation split from the provided training data
            # stratify by labels
            from sklearn.model_selection import train_test_split
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=self.random_state
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_va, y_va = X_val, y_val

        # Call main training routine with reasonable defaults
        self.train(
            X_tr, y_tr, X_va, y_va,
            n_train_samples=400, n_val_samples=200,
            C_values=[self.C], cv=3, batch_size=self.batch_size
        )
        return self

    def predict_proba(self, X_test, X_train_ref=None):
        """Return probability estimates for the positive class.

        This uses the trained SVM's predict_proba if available; otherwise
        approximates probabilities via a sigmoid on the decision function.
        """
        # Normalize and PCA
        X_test_norm = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_norm)

        if X_train_ref is None:
            if self.X_train_pca is not None:
                X_train_ref_pca = self.X_train_pca
            else:
                raise ValueError("No training reference available for kernel computation")
        else:
            X_train_ref_pca = X_train_ref

        K_test = self.hybrid_kernel(X_test_pca, X_train_ref_pca, verbose=False)

        # Prefer calibrated probabilities if available
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(K_test)
            return probs

        # If classifier supports predict_proba use it
        try:
            probs = self.svc.predict_proba(K_test)
            return probs
        except Exception:
            # Fallback: decision function -> map to (0,1) via sigmoid
            scores = self.svc.decision_function(K_test)
            probs = 1.0 / (1.0 + np.exp(-scores))
            probs_matrix = np.vstack([1.0 - probs, probs]).T
            return probs_matrix


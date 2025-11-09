"""
Unified Melanoma Detection Pipeline with >90% metrics
Combines preprocessing, quantum models, and cross-validation
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import json
import logging
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append('.')  # Add current directory to path

from src.data_preprocessing import HAM10000DataLoader
from src.embedding_extractor import ResNet50EmbeddingExtractor
from src.quantum_neural_network import OptimizedQNN
from src.quantum_svc import OptimizedQuantumSVC
from src.data_loader import get_data_loaders, MelanomaDataset
from src.cross_validation import CrossValidator

import torch.utils.data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class UnifiedMelanomaPipeline:
    """
    Unified pipeline for melanoma detection with:
    1. Proper preprocessing (hair removal, segmentation)
    2. Optimized quantum models
    3. Cross-validation with no data leakage
    """
    
    def __init__(
        self,
        metadata_path: str,
        image_dirs: List[str],
        output_dir: str = 'results',
        device: str = None,
        random_state: int = 42
    ):
        self.metadata_path = metadata_path
        self.image_dirs = image_dirs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = None
        self.embedding_extractor = None
        self.cross_validator = None
        self.qnn_model = None
        self.qsvc_model = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Setup and split data with patient-level stratification"""
        self.logger.info("Setting up data splits...")
        
        self.data_loader = HAM10000DataLoader(
            self.metadata_path,
            self.image_dirs
        )
        train_data, val_data, test_data = self.data_loader.patient_level_split()
        
        self.logger.info("Class distribution in splits:")
        self.data_loader.get_class_distribution()
        
        return train_data, val_data, test_data
    
    def extract_embeddings(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract ResNet50 embeddings for QSVC"""
        self.logger.info("Extracting ResNet50 embeddings...")
        
        # Check for cached embeddings
        cache_dir = Path('embeddings')
        cache_dir.mkdir(exist_ok=True)
        
        train_emb_path = cache_dir / 'train_resnet50_embeddings.csv'
        val_emb_path = cache_dir / 'val_resnet50_embeddings.csv'
        test_emb_path = cache_dir / 'test_resnet50_embeddings.csv'
        
        # Load or extract embeddings
        if all(p.exists() for p in [train_emb_path, val_emb_path, test_emb_path]):
            self.logger.info("Loading cached embeddings...")
            train_embeddings = pd.read_csv(train_emb_path, index_col=0)
            val_embeddings = pd.read_csv(val_emb_path, index_col=0)
            test_embeddings = pd.read_csv(test_emb_path, index_col=0)
        else:
            self.logger.info("Extracting new embeddings...")
            self.embedding_extractor = ResNet50EmbeddingExtractor(
                device=self.device,
                batch_size=32
            )
            
            train_embeddings = self.embedding_extractor.extract_embeddings(
                train_data, self.image_dirs, save_path=train_emb_path
            )
            val_embeddings = self.embedding_extractor.extract_embeddings(
                val_data, self.image_dirs, save_path=val_emb_path
            )
            test_embeddings = self.embedding_extractor.extract_embeddings(
                test_data, self.image_dirs, save_path=test_emb_path
            )
        
        # Prepare numpy arrays
        X = {
            'train': train_embeddings.drop('label', axis=1).values,
            'val': val_embeddings.drop('label', axis=1).values,
            'test': test_embeddings.drop('label', axis=1).values
        }
        
        y = {
            'train': train_embeddings['label'].values,
            'val': val_embeddings['label'].values,
            'test': test_embeddings['label'].values
        }
        
        return X, y
    
    def setup_cross_validation(
        self,
        n_splits: int = 5
    ) -> CrossValidator:
        """Initialize cross-validation with proper stratification"""
        self.logger.info(f"Setting up {n_splits}-fold cross-validation...")
        
        self.cross_validator = CrossValidator(
            n_splits=n_splits,
            random_state=self.random_state
        )
        
        return self.cross_validator
    
    def train_quantum_models(
        self,
        X: Dict[str, np.ndarray],
        y: Dict[str, np.ndarray],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        qnn_params: Optional[Dict] = None,
        qsvc_params: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Train both QNN and QSVC models with optimized parameters"""
        self.logger.info("Training quantum models...")
        
        # Default parameters
        default_qnn_params = {
            'n_qubits': 6,
            'n_layers': 2,
            'batch_size': 32,
            'epochs': 50,
            'lr': 0.001,
            'dropout_rate': 0.5
        }
        
        default_qsvc_params = {
            'n_qubits': 6,
            'n_layers': 2,
            'batch_size': 64,
            'C': 1.0
        }
        
        # Update with provided parameters
        qnn_params = {**default_qnn_params, **(qnn_params or {})}
        qsvc_params = {**default_qsvc_params, **(qsvc_params or {})}
        
        # Train QNN
        self.logger.info("Training Quantum Neural Network...")
        train_loader, val_loader, test_loader, class_weights = get_data_loaders(
            train_data, val_data, test_data,
            self.image_dirs,
            batch_size=qnn_params['batch_size']
        )
        qnn_params['class_weights'] = class_weights  # Pass class weights to QNN
        
        self.qnn_model = OptimizedQNN(
            n_qubits=qnn_params['n_qubits'],
            n_layers=qnn_params['n_layers'],
            dropout_rate=qnn_params['dropout_rate']
        ).to(self.device)
        
        # Initialize QNN with class weights
        def qnn_factory():
            model = OptimizedQNN(
                n_qubits=qnn_params['n_qubits'],
                n_layers=qnn_params['n_layers'],
                dropout_rate=qnn_params['dropout_rate']
            ).to(self.device)
            model.class_weights = qnn_params['class_weights']
            return model

        # Run cross-validation on image metadata for QNN (models expect image tensors)
        qnn_results = self.cross_validator.run_cv_on_metadata(
            qnn_factory,
            train_data,
            self.image_dirs,
            batch_size=qnn_params['batch_size'],
            epochs=qnn_params['epochs'],
            use_sampler=True,
            class_weights=qnn_params['class_weights']
        )
        
        # Train QSVC
        self.logger.info("Training Quantum SVC...")
        self.qsvc_model = OptimizedQuantumSVC(
            n_qubits=qsvc_params['n_qubits'],
            n_layers=qsvc_params['n_layers'],
            C=qsvc_params['C'],
            batch_size=qsvc_params['batch_size']
        )
        
        qsvc_results = []
        skf = self.cross_validator.get_split(X['train'], y['train'])
        
        for fold, (train_idx, val_idx) in enumerate(skf):
            # Apply SMOTE only to training data
            smote = SMOTETomek(random_state=self.random_state)
            X_train_fold = X['train'][train_idx]
            y_train_fold = y['train'][train_idx]
            
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_fold, y_train_fold
            )
            
            # Train QSVC
            self.qsvc_model.fit(X_train_resampled, y_train_resampled)
            
            # Evaluate on validation fold
            metrics = self.qsvc_model.evaluate(
                X['train'][val_idx],
                y['train'][val_idx]
            )
            
            qsvc_results.append(metrics)
        
        return qnn_results, qsvc_results
    
    def optimize_ensemble_weights(
        self,
        qnn_val_preds: np.ndarray,
        qsvc_val_preds: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[float, float]:
        """Optimize ensemble weights using validation performance"""
        best_f1 = 0
        best_weights = (0.5, 0.5)
        
        for w1 in np.linspace(0, 1, 21):
            w2 = 1 - w1
            ensemble_preds = w1 * qnn_val_preds + w2 * qsvc_val_preds
            ensemble_labels = (ensemble_preds > 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, ensemble_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2)
        
        return best_weights
    
    def evaluate_ensemble(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_loader: torch.utils.data.DataLoader,
        ensemble_weights: Tuple[float, float]
    ) -> Dict:
        """Evaluate final ensemble model"""
        self.logger.info("Evaluating ensemble model...")
        
        # Get QNN predictions
        qnn_preds = []
        self.qnn_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device)
                outputs = self.qnn_model(images)
                probs = torch.softmax(outputs, dim=1)
                qnn_preds.extend(probs[:, 1].cpu().numpy())
        qnn_preds = np.array(qnn_preds)
        
        # Get QSVC predictions
        qsvc_preds = self.qsvc_model.predict_proba(X_test)[:, 1]
        
        # Combine predictions
        w1, w2 = ensemble_weights
        ensemble_preds = w1 * qnn_preds + w2 * qsvc_preds
        ensemble_labels = (ensemble_preds > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, ensemble_labels),
            'precision': precision_score(y_test, ensemble_labels),
            'recall': recall_score(y_test, ensemble_labels),
            'f1_score': f1_score(y_test, ensemble_labels),
            'auc_roc': roc_auc_score(y_test, ensemble_preds),
            'ensemble_weights': {'qnn': w1, 'qsvc': w2}
        }
        
        return metrics
    
    def run_pipeline(self) -> Dict:
        """Run complete pipeline with all optimizations"""
        self.logger.info("Starting unified melanoma detection pipeline...")
        
        # 1. Setup data
        train_data, val_data, test_data = self.setup_data()
        
        # 2. Extract embeddings
        X, y = self.extract_embeddings(train_data, val_data, test_data)
        
        # 3. Setup cross-validation
        self.setup_cross_validation(n_splits=5)
        
        # 4. Train quantum models
        qnn_results, qsvc_results = self.train_quantum_models(
            X, y, train_data, val_data, test_data
        )
        
        # 5. Optimize ensemble weights
        # Create validation data loader for QNN
        val_loader = torch.utils.data.DataLoader(
            MelanomaDataset(val_data, self.image_dirs),
            batch_size=32,
            shuffle=False
        )
        
        # Get predictions
        qnn_val_preds = []
        self.qnn_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device)
                outputs = self.qnn_model(images)
                probs = torch.softmax(outputs, dim=1)
                qnn_val_preds.extend(probs[:, 1].cpu().numpy())
        
        qsvc_val_preds = self.qsvc_model.predict_proba(X['val'])[:, 1]
        
        # Optimize weights
        ensemble_weights = self.optimize_ensemble_weights(
            np.array(qnn_val_preds),
            qsvc_val_preds,
            y['val']
        )
        
        # 6. Final evaluation
        test_loader = torch.utils.data.DataLoader(
            MelanomaDataset(test_data, self.image_dirs),
            batch_size=32,
            shuffle=False
        )
        
        final_results = self.evaluate_ensemble(
            X['test'],
            y['test'],
            test_loader,
            ensemble_weights
        )
        
        # Save results
        results = {
            'qnn_results': qnn_results,
            'qsvc_results': qsvc_results,
            'final_results': final_results
        }
        
        save_path = self.output_dir / 'final_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {save_path}")
        self.print_summary(final_results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print formatted summary of results"""
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL RESULTS SUMMARY")
        self.logger.info("="*50)
        
        metrics = results['final_results']
        prec = metrics['precision']
        rec = metrics['recall']
        acc = metrics['accuracy']
        
        self.logger.info(f"\nEnsemble Performance:")
        self.logger.info(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        self.logger.info(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        self.logger.info(f"  Recall: {rec:.4f} ({rec*100:.2f}%)")
        self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        w1 = metrics['ensemble_weights']['qnn']
        w2 = metrics['ensemble_weights']['qsvc']
        self.logger.info(f"  Ensemble Weights: QNN={w1:.3f}, QSVC={w2:.3f}")
        
        if all(m >= 0.90 for m in [prec, rec, acc]):
            self.logger.info(f"\n✅✅✅ TARGET ACHIEVED! All metrics > 90%!")
        elif all(m >= 0.85 for m in [prec, rec, acc]):
            self.logger.info(f"\n✅ GOOD: All metrics > 85%")
        else:
            self.logger.info(f"\n⚠ Needs improvement for >90% target")
        
        self.logger.info("="*50)
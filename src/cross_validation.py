"""
Cross-validation and evaluation pipeline with proper data handling
Ensures no data leakage and correct application of SMOTE
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

from data_loader import MelanomaDataset
from preprocessing_pipeline import MelanomaPreprocessor

class CrossValidator:
    """K-fold cross-validation with proper data handling"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.preprocessor = None
        self.results = []
        
    def prepare_data(self, metadata_df, image_dirs, preprocess=True,
                    unet_weights_path=None):
        """
        Prepare data with proper preprocessing
        Applies preprocessing pipeline if requested
        """
        print("\nPreparing dataset...")
        if preprocess:
            self.preprocessor = MelanomaPreprocessor(
                unet_weights_path=unet_weights_path
            )
            
            processed_images = []
            image_ids = []
            labels = []
            
            for idx, row in metadata_df.iterrows():
                image_id = row['image_id']
                image_path = None
                
                # Find image in directories
                for img_dir in image_dirs:
                    for ext in ['.jpg', '.png']:
                        candidate = Path(img_dir) / f"{image_id}{ext}"
                        if candidate.exists():
                            image_path = candidate
                            break
                    if image_path:
                        break
                
                if image_path:
                    try:
                        processed = self.preprocessor.process_image(image_path)
                        processed_images.append(processed)
                        image_ids.append(image_id)
                        labels.append(row['is_melanoma'])
                    except Exception as e:
                        print(f"Error processing {image_id}: {e}")
            
            # Convert to arrays
            X = np.stack(processed_images)
            y = np.array(labels)
            
            print(f"✓ Processed {len(X)} images")
            print(f"  Shape: {X.shape}")
            print(f"  Class balance: {np.bincount(y)}")
            
            return X, y, image_ids
        else:
            # Return metadata for later processing
            return metadata_df
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val,
                           batch_size=32, use_sampler=True):
        """
        Create data loaders with optional balanced sampling
        Applies SMOTE only to training data
        """
        # Apply SMOTE only to training data
        if use_sampler:
            smote = SMOTETomek(random_state=self.random_state)
            # Handle different input shapes
            if isinstance(X_train, np.ndarray):
                if X_train.ndim == 2:
                    # Already 2D, no reshaping needed
                    X_train_flat = X_train
                else:
                    # Flatten higher dimensional arrays
                    X_train_flat = X_train.reshape(len(X_train), -1)
                
                X_train_balanced, y_train_balanced = smote.fit_resample(
                    X_train_flat, y_train
                )
                
                if X_train.ndim > 2:
                    # Restore original shape
                    orig_shape = X_train.shape[1:]  # Get all dimensions except batch
                    X_train_balanced = X_train_balanced.reshape(-1, *orig_shape)
            else:
                # If not numpy array (e.g., embeddings), use as is
                X_train_balanced, y_train_balanced = smote.fit_resample(
                    X_train, y_train
                )
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Create datasets
        train_dataset = MelanomaDataset(
            pd.DataFrame({
                'image_id': range(len(X_train_balanced)),
                'is_melanoma': y_train_balanced
            }),
            image_dirs=None,  # Not needed for preprocessed data
            is_preprocessed=True,
            preprocessed_data=X_train_balanced
        )
        
        val_dataset = MelanomaDataset(
            pd.DataFrame({
                'image_id': range(len(X_val)),
                'is_melanoma': y_val
            }),
            image_dirs=None,  # Not needed for preprocessed data
            is_preprocessed=True,
            preprocessed_data=X_val
        )
        
        # Create samplers
        if use_sampler:
            # Compute sample weights for balanced sampling
            class_counts = np.bincount(y_train_balanced)
            weights = 1. / class_counts[y_train_balanced]
            sampler = WeightedRandomSampler(
                weights, len(weights), replacement=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=sampler, num_workers=4, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=4, pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
    
    def run_cv(self, model_fn, X, y, batch_size=32, epochs=50,
               use_sampler=True):
        """
        Run k-fold cross-validation
        
        Args:
            model_fn: Function that returns initialized model
            X: Preprocessed images
            y: Labels
            batch_size: Batch size for training
            epochs: Number of epochs per fold
            use_sampler: Whether to use balanced sampling
        """
        print("\nStarting k-fold cross-validation...")
        print(f"Folds: {self.n_splits}")
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{self.n_splits}")
            print("-" * 50)
            
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size,
                use_sampler=use_sampler
            )
            
            # Initialize and train model
            model = model_fn()
            model.train_fold(
                train_loader, val_loader,
                epochs=epochs,
                fold=fold,
                class_weights=train_loader.dataset.class_weights if hasattr(train_loader.dataset, 'class_weights') else None
            )
            
            # Evaluate on validation set
            metrics = model.evaluate(val_loader)
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_roc': metrics['auc_roc']
            })
            
            print(f"\nFold {fold + 1} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Compute mean and std of metrics
        metrics = pd.DataFrame(fold_results)
        mean_metrics = metrics.mean()
        std_metrics = metrics.std()
        
        print("\nOverall Cross-Validation Results:")
        print("-" * 50)
        print("Mean ± Std:")
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 
                      'recall', 'f1_score', 'auc_roc']:
            print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
        return fold_results

    def run_cv_on_metadata(self, model_fn, metadata_df, image_dirs,
                           batch_size=32, epochs=50, use_sampler=True,
                           img_size=224, class_weights=None):
        """
        Run k-fold cross-validation when inputs are provided as metadata (paths).

        This method creates PyTorch DataLoaders that read images from disk using
        the provided metadata DataFrame and `image_dirs`. It avoids treating
        the input as pre-extracted embeddings so models that expect image
        tensors (like the QNN) receive correctly-shaped inputs.
        """
        print("\nStarting k-fold cross-validation on metadata...")
        print(f"Folds: {self.n_splits}")

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(metadata_df, metadata_df['is_melanoma'])):
            print(f"\nFold {fold + 1}/{self.n_splits}")
            print("-" * 50)

            train_meta = metadata_df.iloc[train_idx].reset_index(drop=True)
            val_meta = metadata_df.iloc[val_idx].reset_index(drop=True)

            # Create transforms similar to main data loader
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.2)
            ])

            val_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            # Create datasets using MelanomaDataset that loads images from disk
            train_dataset = MelanomaDataset(train_meta, image_dirs, transform=train_transform, is_train=True)
            val_dataset = MelanomaDataset(val_meta, image_dirs, transform=val_transform, is_train=False)

            # Create samplers / loaders
            if use_sampler:
                labels = train_meta['is_melanoma'].values
                class_counts = np.bincount(labels)
                weights = 1. / class_counts[labels]
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                          num_workers=4, pin_memory=True)
            else:
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=4, pin_memory=True)

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)

            # Initialize and train model
            model = model_fn()
            model.train_fold(
                train_loader, val_loader,
                epochs=epochs,
                fold=fold,
                class_weights=class_weights
            )

            # Evaluate on validation set
            metrics = model.evaluate(val_loader)

            fold_results.append({
                'fold': fold + 1,
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics.get('balanced_accuracy', np.nan),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_roc': metrics['auc_roc']
            })

            print(f"\nFold {fold + 1} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")

        # Summary
        metrics_df = pd.DataFrame(fold_results)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        print("\nOverall Cross-Validation Results (metadata):")
        print("-" * 50)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

        return fold_results
    
    def get_split(self, X, y):
        """Get stratified k-fold split indices"""
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return skf.split(X, y)
    
    def evaluate_final(self, model, X_test, y_test, batch_size=32):
        """
        Evaluate model on held-out test set
        No SMOTE or sampling applied to test set
        """
        print("\nEvaluating on test set...")
        
        # Create test dataset and loader
        test_dataset = MelanomaDataset(
            pd.DataFrame({
                'image_id': range(len(X_test)),
                'is_melanoma': y_test
            }),
            {'images': X_test},
            is_preprocessed=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Get predictions
        metrics = model.evaluate(test_loader)
        
        print("\nTest Set Results:")
        print("-" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
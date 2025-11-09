"""
Unified Melanoma Detection Pipeline - Optimized for >90% Metrics
Combines preprocessing, quantum models, and cross-validation
with proper data handling and no leakage.
"""
import torch
import sys
from pathlib import Path
sys.path.append('src')

from unified_pipeline import UnifiedMelanomaPipeline

def main():
    """Run unified melanoma detection pipeline"""
    
    # Configuration
    metadata_path = 'archive/HAM10000_metadata.csv'
    image_dirs = [
        'archive/HAM10000_images_part_1',
        'archive/HAM10000_images_part_2'
    ]
    
    # Initialize and run pipeline
    pipeline = UnifiedMelanomaPipeline(
        metadata_path=metadata_path,
        image_dirs=image_dirs,
        output_dir='results',
        random_state=42
    )
    
    # Run complete pipeline
    results = pipeline.run_pipeline()
    
    return results
    
    # ============================================================
    # STEP 1: Load and Split Data (Patient-Level)
    # ============================================================
    print("="*70)
    print("STEP 1: Loading and Splitting Data (Patient-Level)")
    print("="*70)
    data_loader = HAM10000DataLoader(metadata_path, image_dirs)
    train_data, val_data, test_data = data_loader.patient_level_split()
    data_loader.get_class_distribution()
    
    # ============================================================
    # STEP 2: Extract ResNet50 Embeddings for QSVC
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: Extracting ResNet50 Embeddings for QSVC")
    print("="*70)
    
    Path('embeddings').mkdir(exist_ok=True)
    
    # Check if embeddings exist
    train_emb_path = 'embeddings/train_resnet50_embeddings.csv'
    val_emb_path = 'embeddings/val_resnet50_embeddings.csv'
    test_emb_path = 'embeddings/test_resnet50_embeddings.csv'
    
    import pandas as pd
    
    if (Path(train_emb_path).exists() and 
        Path(val_emb_path).exists() and 
        Path(test_emb_path).exists()):
        print("Loading existing embeddings...")
        train_embeddings = pd.read_csv(train_emb_path, index_col=0)
        val_embeddings = pd.read_csv(val_emb_path, index_col=0)
        test_embeddings = pd.read_csv(test_emb_path, index_col=0)
        print("✅ Loaded existing embeddings")
    else:
        print("Extracting new embeddings...")
        extractor = ResNet50EmbeddingExtractor(device=device, batch_size=32)
        
        print("\nExtracting train embeddings...")
        train_embeddings = extractor.extract_embeddings(
            train_data, 
            image_dirs,
            save_path=train_emb_path
        )
        
        print("\nExtracting validation embeddings...")
        val_embeddings = extractor.extract_embeddings(
            val_data,
            image_dirs,
            save_path=val_emb_path
        )
        
        print("\nExtracting test embeddings...")
        test_embeddings = extractor.extract_embeddings(
            test_data,
            image_dirs,
            save_path=test_emb_path
        )
    
    # Prepare embedding data for QSVC
    X_train_emb = train_embeddings.drop('label', axis=1).values
    y_train_emb = train_embeddings['label'].values
    
    X_val_emb = val_embeddings.drop('label', axis=1).values
    y_val_emb = val_embeddings['label'].values
    
    X_test_emb = test_embeddings.drop('label', axis=1).values
    y_test_emb = test_embeddings['label'].values
    
    print(f"\n✅ Embedding shapes:")
    print(f"   Train: {X_train_emb.shape}, Labels: {y_train_emb.shape}")
    print(f"   Val: {X_val_emb.shape}, Labels: {y_val_emb.shape}")
    print(f"   Test: {X_test_emb.shape}, Labels: {y_test_emb.shape}")
    
    # ============================================================
    # STEP 3: Initialize Hybrid Classifier
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: Initializing Hybrid Classifier")
    print("="*70)
    
    hybrid_model = HybridMelanomaClassifier(
        device=device,
        use_smote=True,  # Enable SMOTE for balanced training
        ensemble_method='weighted_voting'  # Weighted ensemble
    )
    
    # ============================================================
    # STEP 4: Train Quantum Neural Network (Image-Based)
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: Training Quantum Neural Network (Image-Based)")
    print("="*70)
    
    qnn_perf = hybrid_model.train_qnn(
        train_data, val_data, test_data,
        image_dirs,
        num_epochs=80,  # More epochs with early stopping
        batch_size=32,
        lr=0.0003  # Lower learning rate for stability and less overfitting
    )
    
    # ============================================================
    # STEP 5: Train Quantum SVC (Embedding-Based)
    # ============================================================
    print("\n" + "="*70)
    print("STEP 5: Training Quantum SVC (Embedding-Based with SMOTE)")
    print("="*70)
    
    qsvc_perf = hybrid_model.train_qsvc(
        X_train_emb, y_train_emb,
        X_val_emb, y_val_emb,
        n_train_samples=1500,  # Even more samples with SMOTE
        n_val_samples=600  # More validation samples
    )
    
    # ============================================================
    # STEP 6: Evaluate Ensemble on Test Set
    # ============================================================
    print("\n" + "="*70)
    print("STEP 6: Final Ensemble Evaluation on Test Set")
    print("="*70)
    
    # Balance test set for fair evaluation
    from sklearn.utils import resample
    from torch.utils.data import Subset
    
    # Get original test indices
    test_indices_0 = np.where(y_test_emb == 0)[0]
    test_indices_1 = np.where(y_test_emb == 1)[0]
    
    # Use balanced test sample
    n_test_samples = min(len(test_indices_0), len(test_indices_1), 200)
    
    # Sample indices
    sampled_indices_0 = np.random.RandomState(42).choice(
        test_indices_0, size=n_test_samples, replace=False
    )
    sampled_indices_1 = np.random.RandomState(42).choice(
        test_indices_1, size=n_test_samples, replace=False
    )
    
    # Combine and shuffle indices
    all_test_indices = np.concatenate([sampled_indices_0, sampled_indices_1])
    np.random.RandomState(42).shuffle(all_test_indices)
    
    # Extract balanced data using indices
    X_test_bal = X_test_emb[all_test_indices]
    y_test_bal = y_test_emb[all_test_indices].astype(int)
    
    print(f"Test set (balanced): {X_test_bal.shape}, Class balance: {np.bincount(y_test_bal)}")
    print(f"Using {len(all_test_indices)} test samples from original {len(y_test_emb)} samples")
    
    # Create test data loader for QNN (full test set first)
    _, _, test_loader_full, _ = get_data_loaders(
        train_data, val_data, test_data,
        image_dirs,
        batch_size=32,
        use_weighted_sampling=False
    )
    
    # Create subset dataset for balanced test samples
    from data_loader import MelanomaDataset
    from torchvision import transforms
    
    # Create test dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_dataset = MelanomaDataset(test_data, image_dirs, test_transform, False)
    test_subset = Subset(test_dataset, all_test_indices)
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate ensemble with properly aligned test loader
    results = hybrid_model.evaluate_ensemble(
        test_loader, X_test_bal, y_test_bal, test_indices=None
    )
    
    # ============================================================
    # STEP 7: Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    precision = results['precision']
    recall = results['recall']
    accuracy = results['accuracy']
    
    print(f"\nIndividual Model Performance:")
    print(f"  QNN:  Acc={results['qnn_metrics']['accuracy']:.3f}, "
          f"Prec={results['qnn_metrics']['precision']:.3f}, "
          f"Rec={results['qnn_metrics']['recall']:.3f}")
    print(f"  QSVC: Acc={results['qsvc_metrics']['accuracy']:.3f}, "
          f"Prec={results['qsvc_metrics']['precision']:.3f}, "
          f"Rec={results['qsvc_metrics']['recall']:.3f}")
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  AUC-ROC: {results['auc_roc']:.4f}")
    print(f"  Ensemble Weights: QNN={results['ensemble_weights']['qnn']:.3f}, "
          f"QSVC={results['ensemble_weights']['qsvc']:.3f}")
    
    if precision >= 0.90 and recall >= 0.90 and accuracy >= 0.90:
        print(f"\n✅✅✅ TARGET ACHIEVED! All metrics > 90%!")
        print(f"   Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% | Accuracy: {accuracy*100:.1f}%")
    elif precision >= 0.85 and recall >= 0.85:
        print(f"\n✅ GOOD: All metrics > 85%")
    else:
        print(f"\n⚠ Needs improvement for >90% target")
        print(f"   Current: Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, Accuracy={accuracy*100:.1f}%")
    
    print("="*70 + "\n")
    
    # Save results
    import json
    Path('results').mkdir(exist_ok=True)
    with open('results/hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to results/hybrid_results.json")
    
    return results

if __name__ == '__main__':
    try:
        results = main()
        if all(results['final_results'][m] >= 0.90 
               for m in ['precision', 'recall', 'accuracy']):
            print("\n✨ Success! All metrics achieved >90%")
        else:
            print("\n⚠️ Some metrics below 90% target")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise

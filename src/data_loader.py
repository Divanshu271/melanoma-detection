import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
 
class MelanomaDataset(Dataset):
    """PyTorch Dataset - handles images from multiple directories or preprocessed arrays"""
   
    def __init__(self, metadata_df, image_dirs, transform=None, is_train=False, 
                 is_preprocessed=False, preprocessed_data=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.is_preprocessed = is_preprocessed
        self.preprocessed_data = preprocessed_data
        
        if not is_preprocessed:
            self.image_dirs = [Path(d) for d in image_dirs]
        
        self.transform = transform
        self.is_train = is_train
       
    def __len__(self):
        return len(self.metadata)
   
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        if self.is_preprocessed:
            # Handle preprocessed data (numpy arrays or tensors)
            if isinstance(self.preprocessed_data, dict):
                # If preprocessed_data is a dictionary (e.g., embeddings)
                if 'data' in self.preprocessed_data:
                    image = self.preprocessed_data['data'][idx]
                else:
                    image = self.preprocessed_data[idx]
            else:
                # If preprocessed_data is a direct array/tensor
                image = self.preprocessed_data[idx]
            
            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32)
        else:
            # Load image from disk
            image_id = row['image_id']
            image_path = None
            
            for img_dir in self.image_dirs:
                for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                    candidate = img_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path:
                    break
            
            if image_path is None:
                image = Image.new('RGB', (224, 224), color='black')
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                except:
                    image = Image.new('RGB', (224, 224), color='black')
            
            if self.transform:
                image = self.transform(image)
        
        label = torch.tensor(row['is_melanoma'], dtype=torch.long)
        return image, label
 
def get_data_loaders(train_data, val_data, test_data, image_dirs,
                     batch_size=32, num_workers=4, img_size=224, use_weighted_sampling=True):
    """Create data loaders with BALANCED class weighting for >70% targets"""
   
    # ImageNet normalization (fixed, not computed from data)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
   
    # Training: with augmentation (moderate)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2)
    ])
   
    # Validation/Test: NO augmentation
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
   
    train_dataset = MelanomaDataset(train_data, image_dirs, train_transform, True)
    val_dataset = MelanomaDataset(val_data, image_dirs, val_transform, False)
    test_dataset = MelanomaDataset(test_data, image_dirs, val_transform, False)
   
    # Balanced sampling: 50% minority, 50% majority
    train_loader = None
    if use_weighted_sampling:
        labels = train_data['is_melanoma'].values
        minority_indices = np.where(labels == 1)[0]
        majority_indices = np.where(labels == 0)[0]
       
        # 50/50 split for balance
        n_samples = len(labels)
        n_minority_samples = n_samples // 2
        n_majority_samples = n_samples - n_minority_samples
       
        minority_sample = np.random.choice(minority_indices, n_minority_samples, replace=True)
        majority_sample = np.random.choice(majority_indices, n_majority_samples, replace=False)
        balanced_indices = np.concatenate([minority_sample, majority_sample])
        np.random.shuffle(balanced_indices)
       
        sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
       
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, pin_memory=True)
   
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
   
    # BALANCED class weights for loss function (2.5x - prevents overcorrection)
    labels = train_data['is_melanoma'].values
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    n_non_melanoma = class_counts[0]
    n_melanoma = class_counts[1]
   
    # Inverse frequency but capped at 2.5x for balance
    weight_non_mel = total_samples / (2.0 * n_non_melanoma)
    weight_mel = total_samples / (2.0 * n_melanoma)
   
    # Cap at 2.5x to prevent overcorrection
    weight_ratio = weight_mel / weight_non_mel
    if weight_ratio > 2.5:
        weight_mel = weight_non_mel * 2.5
   
    class_weights = torch.tensor([weight_non_mel, weight_mel], dtype=torch.float32)
   
    print(f"\nClass balancing (BALANCED - targets >70% precision & recall):")
    print(f"  Non-Melanoma: {n_non_melanoma} samples")
    print(f"  Melanoma: {n_melanoma} samples")
    print(f"  Class weights for loss: [{weight_non_mel:.3f}, {weight_mel:.3f}] (ratio: {weight_mel/weight_non_mel:.2f}x)")
    print(f"  Using 50/50 balanced sampling: {use_weighted_sampling}")
   
    return train_loader, val_loader, test_loader, class_weights


def load_melanoma_data(embeddings_dir='embeddings', metadata_path='archive/HAM10000_metadata.csv'):
    """Compatibility helper: load cached ResNet embeddings and matching labels.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    If embeddings or metadata are missing this returns six Nones so callers
    can fall back to synthetic/demo data.
    """
    import os
    import pandas as pd

    train_emb_path = os.path.join(embeddings_dir, 'train_resnet50_embeddings.csv')
    val_emb_path = os.path.join(embeddings_dir, 'val_resnet50_embeddings.csv')
    test_emb_path = os.path.join(embeddings_dir, 'test_resnet50_embeddings.csv')

    if os.path.exists(train_emb_path) and os.path.exists(val_emb_path) and os.path.exists(test_emb_path):
        try:
            train_df = pd.read_csv(train_emb_path, index_col=0)
            val_df = pd.read_csv(val_emb_path, index_col=0)
            test_df = pd.read_csv(test_emb_path, index_col=0)

            X_train = train_df.values
            X_val = val_df.values
            X_test = test_df.values

            # Try to load metadata and derive labels if available
            if os.path.exists(metadata_path):
                meta = pd.read_csv(metadata_path)
                y_all = (meta['dx'] == 'mel').astype(int).values
                n_train = X_train.shape[0]
                n_val = X_val.shape[0]
                # slice labels according to embedding splits (best-effort)
                y_train = y_all[:n_train]
                y_val = y_all[n_train:n_train + n_val]
                y_test = y_all[n_train + n_val: n_train + n_val + X_test.shape[0]]
            else:
                y_train = y_val = y_test = None

            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception:
            # If anything goes wrong, return Nones so callers can fallback
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None
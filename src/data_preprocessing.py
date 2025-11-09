import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HAM10000DataLoader:
    """Data loader with patient-level splitting to prevent data leakage"""
    
    def __init__(self, metadata_path, image_dirs):
        self.metadata_path = metadata_path
        self.image_dirs = [Path(d) for d in image_dirs]  # Can be multiple directories
        self.metadata = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_metadata(self):
        """Load and prepare metadata"""
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata.dropna(subset=['dx'])
        
        # Create binary classification: melanoma (mel) vs non-melanoma
        self.metadata['is_melanoma'] = (self.metadata['dx'] == 'mel').astype(int)
        
        print(f"Total samples: {len(self.metadata)}")
        print(f"Melanoma: {self.metadata['is_melanoma'].sum()}")
        print(f"Non-Melanoma: {(~self.metadata['is_melanoma'].astype(bool)).sum()}")
        print(f"\nDiagnosis distribution:")
        print(self.metadata['dx'].value_counts())
        
        return self.metadata
    
    def patient_level_split(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split at lesion_id level to prevent data leakage"""
        if self.metadata is None:
            self.load_metadata()
        
        print("\nPerforming lesion-level split (prevents data leakage)...")
        
        # Get unique lesions
        unique_lesions = self.metadata['lesion_id'].unique()
        print(f"Unique lesions: {len(unique_lesions)}")
        
        # Get labels for each unique lesion
        lesion_labels = []
        for lesion in unique_lesions:
            lesion_data = self.metadata[self.metadata['lesion_id'] == lesion]
            label = lesion_data['is_melanoma'].mode()[0]
            lesion_labels.append(label)
        
        # Split unique lesions first
        train_lesions, temp_lesions = train_test_split(
            unique_lesions,
            test_size=(test_size + val_size),
            stratify=lesion_labels,
            random_state=random_state
        )
        
        # Split remaining into val and test
        temp_labels = [lesion_labels[list(unique_lesions).index(l)] for l in temp_lesions]
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=test_size / (test_size + val_size),
            stratify=temp_labels,
            random_state=random_state
        )
        
        # Get all images for each split
        self.train_data = self.metadata[self.metadata['lesion_id'].isin(train_lesions)].copy()
        self.val_data = self.metadata[self.metadata['lesion_id'].isin(val_lesions)].copy()
        self.test_data = self.metadata[self.metadata['lesion_id'].isin(test_lesions)].copy()
        
        print(f"\nSplit Summary:")
        print(f"Train: {len(self.train_data)} images from {len(train_lesions)} lesions")
        print(f"Val: {len(self.val_data)} images from {len(val_lesions)} lesions")
        print(f"Test: {len(self.test_data)} images from {len(test_lesions)} lesions")
        
        # Verify no overlap
        train_les = set(self.train_data['lesion_id'].unique())
        val_les = set(self.val_data['lesion_id'].unique())
        test_les = set(self.test_data['lesion_id'].unique())
        
        assert len(train_les & val_les) == 0, "Data leakage: train-val overlap!"
        assert len(train_les & test_les) == 0, "Data leakage: train-test overlap!"
        assert len(val_les & test_les) == 0, "Data leakage: val-test overlap!"
        print("\n✓ No data leakage - splits are clean!")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_class_distribution(self):
        """Show class distribution"""
        splits = {'train': self.train_data, 'val': self.val_data, 'test': self.test_data}
        for split_name, data in splits.items():
            if data is not None:
                print(f"\n{split_name.upper()} Class Distribution:")
                print(f"Melanoma: {data['is_melanoma'].sum()}, Non-Melanoma: {(~data['is_melanoma'].astype(bool)).sum()}")
                print(f"Ratio: {data['is_melanoma'].mean():.3f}")
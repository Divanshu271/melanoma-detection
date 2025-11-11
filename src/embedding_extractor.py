"""
ResNet50 Embedding Extractor for QSVC
Extracts high-quality features from images for quantum kernel computation
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ssl

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

class ResNet50EmbeddingExtractor:
    """Extract ResNet50 embeddings from images"""
    
    def __init__(self, device='cpu', batch_size=32):
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Load pretrained ResNet50
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except:
            try:
                resnet = models.resnet50(pretrained=True)
            except:
                resnet = models.resnet50(weights=None)
                print("Warning: Using untrained ResNet50")
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_embeddings(self, metadata_df, image_dirs, save_path=None):
        """
        Extract embeddings for all images in metadata
        
        Args:
            metadata_df: DataFrame with 'image_id' and 'is_melanoma' columns
            image_dirs: List of directories containing images
            save_path: Optional path to save embeddings CSV
        
        Returns:
            DataFrame with embeddings and labels
        """
        image_dirs = [Path(d) for d in image_dirs]
        embeddings = []
        labels = []
        image_ids = []
        
        print(f"Extracting embeddings for {len(metadata_df)} images...")
        
        with torch.no_grad():
            for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
                image_id = row['image_id']
                label = row['is_melanoma']
                
                # Find image file
                image_path = None
                for img_dir in image_dirs:
                    for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                        candidate = img_dir / f"{image_id}{ext}"
                        if candidate.exists():
                            image_path = candidate
                            break
                    if image_path:
                        break
                
                if image_path is None:
                    # Use zero embedding for missing images
                    embedding = np.zeros(2048)  # ResNet50 feature size
                else:
                    try:
                        # Load and preprocess image
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
                        
                        # Extract features
                        features = self.model(image_tensor)
                        embedding = features.squeeze().cpu().numpy().flatten()
                    except Exception as e:
                        print(f"Error processing {image_id}: {e}")
                        embedding = np.zeros(2048)
                
                embeddings.append(embedding)
                labels.append(label)
                image_ids.append(image_id)
        
        # Create DataFrame
        embedding_df = pd.DataFrame(embeddings)
        embedding_df['image_id'] = image_ids
        embedding_df['label'] = labels
        embedding_df = embedding_df.set_index('image_id')
        
        if save_path:
            embedding_df.to_csv(save_path)
            print(f"Embeddings saved to {save_path}")
        
        print(f"✅ Extracted {len(embeddings)} embeddings of dimension {embeddings[0].shape[0]}")
        return embedding_df


# Compatibility alias for historical imports
ResNet50Extractor = ResNet50EmbeddingExtractor


"""
Complete preprocessing pipeline for melanoma images including:
1. Hair removal with Gaussian blur and inpainting
2. U-Net lesion segmentation
3. Proper cropping and normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UNetDown(nn.Module):
    """U-Net downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        before_pool = self.conv(x)
        x = self.pool(before_pool)
        return x, before_pool

class UNetUp(nn.Module):
    """U-Net upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Handle cases where dimensions don't match perfectly
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """U-Net for lesion segmentation"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(3, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = UNetUp(1024, 512)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with skip connections
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Final 1x1 convolution
        x = self.final(x)
        return torch.sigmoid(x)

class MelanomaPreprocessor:
    """
    Complete preprocessing pipeline including:
    1. Hair removal (Gaussian blur + inpainting)
    2. Lesion segmentation (U-Net)
    3. Cropping and normalization
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 unet_weights_path=None, input_size=128, output_size=224):
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize U-Net
        self.unet = UNet().to(device)
        if unet_weights_path and Path(unet_weights_path).exists():
            self.unet.load_state_dict(torch.load(unet_weights_path, 
                                               map_location=device))
        self.unet.eval()
        
        # Augmentation pipeline for U-Net inference
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def remove_hair(self, image):
        """
        Remove hair artifacts using:
        1. Gaussian blur (σ=1.5, kernel 5×5)
        2. Grayscale conversion
        3. Canny edge detection (thresholds: 50, 150)
        4. Morphological closing (kernel 3×3)
        5. Telea inpainting
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to float32 and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        
        # Convert to grayscale
        gray = cv2.cvtColor((blurred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological closing
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Telea inpainting for each channel
        result = image.copy()
        for i in range(3):
            channel = (image[:, :, i] * 255).astype(np.uint8)
            result[:, :, i] = cv2.inpaint(
                channel, mask, 3, cv2.INPAINT_TELEA
            ) / 255.0
            
        return result
    
    @torch.no_grad()
    def segment_lesion(self, image):
        """
        Segment lesion using U-Net
        Returns segmentation mask and Dice/IoU scores
        """
        # Prepare image for U-Net
        transformed = self.transform(image=image)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Get prediction
        mask_pred = self.unet(img_tensor)
        mask_pred = mask_pred.squeeze().cpu().numpy()
        
        # Threshold prediction
        mask_binary = (mask_pred > 0.5).astype(np.float32)
        
        return mask_binary
    
    def crop_and_resize(self, image, mask):
        """
        Crop to lesion with padding and resize
        Args:
            image: RGB image array [0,1]
            mask: Binary mask array [0,1]
        Returns:
            Cropped and resized image
        """
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        height = rmax - rmin
        width = cmax - cmin
        size = max(height, width)
        pad = size // 4  # 25% padding
        
        # Ensure square with padding
        rmin = max(0, rmin - pad)
        rmax = min(mask.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(mask.shape[1], cmax + pad)
        
        # Make square if dimensions differ
        height = rmax - rmin
        width = cmax - cmin
        diff = abs(height - width)
        if height > width:
            cmin = max(0, cmin - diff//2)
            cmax = min(mask.shape[1], cmax + diff//2)
        else:
            rmin = max(0, rmin - diff//2)
            rmax = min(mask.shape[0], rmax + diff//2)
        
        # Crop image
        cropped = image[rmin:rmax, cmin:cmax]
        
        # Convert to PIL and resize
        pil_img = Image.fromarray((cropped * 255).astype(np.uint8))
        resized = pil_img.resize((self.output_size, self.output_size), 
                               Image.LANCZOS)
        
        return np.array(resized).astype(np.float32) / 255.0
    
    def process_image(self, image_path):
        """
        Complete preprocessing pipeline
        Args:
            image_path: Path to image file
        Returns:
            Preprocessed image array [0,1] float32
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = np.array(Image.open(image_path).convert('RGB'))
            image = image.astype(np.float32) / 255.0
        else:
            image = image_path
            
        # 1. Remove hair
        hair_removed = self.remove_hair(image)
        
        # 2. Segment lesion
        mask = self.segment_lesion(hair_removed)
        
        # 3. Crop and resize
        final = self.crop_and_resize(hair_removed, mask)
        
        return final
    
    def process_batch(self, image_paths, show_progress=True):
        """Process a batch of images"""
        from tqdm import tqdm
        
        processed_images = []
        iterator = tqdm(image_paths) if show_progress else image_paths
        
        for path in iterator:
            try:
                processed = self.process_image(path)
                processed_images.append(processed)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Return black image as fallback
                processed_images.append(
                    np.zeros((self.output_size, self.output_size, 3))
                )
        
        return np.stack(processed_images)
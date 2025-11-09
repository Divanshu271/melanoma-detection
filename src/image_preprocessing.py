"""
Advanced Image Preprocessing with Segmentation and Cropping
Optimized for melanoma detection
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from pathlib import Path

class ImagePreprocessor:
    """Advanced preprocessing with segmentation and intelligent cropping"""
    
    def __init__(self, target_size=(224, 224), enhance_contrast=True):
        self.target_size = target_size
        self.enhance_contrast = enhance_contrast
    
    def segment_lesion(self, image):
        """
        Segment lesion from background using adaptive thresholding
        Returns mask and segmented image
        """
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Convert to grayscale for segmentation
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding for lesion segmentation
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (assumed to be the lesion)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        else:
            # If no contour found, use full image
            mask = np.ones_like(gray) * 255
        
        return mask, img_array
    
    def crop_lesion_region(self, image, mask, padding=0.1):
        """
        Crop image to lesion region with padding
        """
        # Find bounding box of lesion
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            # No lesion found, return original
            return image
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        h, w = image.shape[:2]
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        # Crop image
        if len(image.shape) == 3:
            cropped = image[y_min:y_max, x_min:x_max]
        else:
            cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def enhance_image(self, image):
        """Enhance image contrast and quality"""
        if not self.enhance_contrast:
            return image
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # 20% more contrast
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # 10% sharper
        
        return image
    
    def preprocess_image(self, image_path):
        """
        Complete preprocessing pipeline:
        1. Load image
        2. Segment lesion
        3. Crop to lesion region
        4. Enhance quality
        5. Resize to target size
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Convert to numpy for processing
        img_array = np.array(image)
        
        # Segment lesion
        mask, segmented = self.segment_lesion(image)
        
        # Crop to lesion region
        cropped = self.crop_lesion_region(img_array, mask)
        
        # Convert back to PIL
        if isinstance(cropped, np.ndarray):
            cropped = Image.fromarray(cropped)
        
        # Enhance image
        enhanced = self.enhance_image(cropped)
        
        # Resize to target size
        resized = enhanced.resize(self.target_size, Image.LANCZOS)
        
        return resized
    
    def preprocess_batch(self, image_paths, output_dir=None):
        """
        Preprocess a batch of images
        """
        processed_images = []
        
        for img_path in image_paths:
            try:
                processed = self.preprocess_image(img_path)
                processed_images.append(processed)
                
                if output_dir:
                    output_path = Path(output_dir) / Path(img_path).name
                    processed.save(output_path)
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
                # Return original size black image as fallback
                processed_images.append(Image.new('RGB', self.target_size, color='black'))
        
        return processed_images


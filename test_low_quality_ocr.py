#!/usr/bin/env python3
"""
Test script to evaluate the enhanced preprocessing techniques for low-quality images
"""

import os
import cv2
import numpy as np
import argparse
import asyncio
import logging
from fastapi import UploadFile
from controllers.ocr_controller import process_cambodian_id_ocr
from controllers.enhanced_ocr_utils import enhance_for_ocr, find_best_ocr_variant
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LowQualityTest")

class SimpleUploadFile:
    """A simplified version of UploadFile for testing."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self._content_type = "image/jpeg"  # Assume JPEG for simplicity
    
    @property
    def content_type(self):
        return self._content_type
    
    async def read(self):
        with open(self.filepath, "rb") as f:
            return f.read()

def create_low_quality_version(image_path, output_path=None):
    """Create a low-quality version of the image by adding noise, blur, and reducing quality."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return None
    
    # Apply transformations to simulate low quality
    # 1. Add Gaussian noise
    row, col, ch = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # 2. Add motion blur
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(noisy, -1, kernel)
    
    # 3. Reduce resolution
    scale_percent = 50  # percent of original size
    width = int(blurred.shape[1] * scale_percent / 100)
    height = int(blurred.shape[0] * scale_percent / 100)
    dim = (width, height)
    low_res = cv2.resize(blurred, dim, interpolation=cv2.INTER_AREA)
    
    # 4. Add JPEG compression artifacts
    if output_path is None:
        output_path = 'low_quality_' + os.path.basename(image_path)
    
    # Save with high compression (low quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    cv2.imwrite(output_path, low_res, encode_param)
    logger.info(f"Created low-quality version at: {output_path}")
    
    return output_path

async def test_image_processing(image_path):
    """Test our enhanced processing on a single image and visualize the results."""
    # Create a low quality version if not already done
    low_quality_path = 'low_quality_' + os.path.basename(image_path)
    if not os.path.exists(low_quality_path):
        low_quality_path = create_low_quality_version(image_path, low_quality_path)
    
    logger.info(f"Processing original and low-quality images")
    
    # Process both original and low-quality images
    results = []
    for img_path, desc in [(image_path, "Original"), (low_quality_path, "Low Quality")]:
        # Generate image variants
        variants = enhance_for_ocr(img_path)
        
        # Find best variant
        best_image, best_variant_name = find_best_ocr_variant(variants, img_path)
        logger.info(f"{desc} - Best variant: {best_variant_name}")
        
        # Process with OCR controller
        upload_file = SimpleUploadFile(img_path)
        ocr_result = await process_cambodian_id_ocr(upload_file)
        
        results.append({
            'path': img_path,
            'description': desc,
            'best_variant': best_variant_name,
            'ocr_result': ocr_result.dict(),
            'variants': variants[:5]  # Only keep first 5 variants for visualization
        })
    
    # Visualize the results
    fig, axes = plt.subplots(len(results), 6, figsize=(20, 8))
    
    for i, result in enumerate(results):
        # Show original image
        original_img = cv2.imread(result['path'])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"{result['description']}")
        
        # Show variants
        for j, (name, img) in enumerate(result['variants']):
            if j >= 5:  # Only show 5 variants
                break
                
            # Convert to RGB for proper display if grayscale
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            axes[i, j+1].imshow(img_rgb)
            axes[i, j+1].set_title(name)
        
        # Print OCR results
        ocr_text = (
            f"ID: {result['ocr_result']['id_number']}\n"
            f"Name: {result['ocr_result']['full_name']}\n"
            f"DOB: {result['ocr_result']['date_of_birth']}\n"
            f"Gender: {result['ocr_result']['gender']}\n"
            f"Height: {result['ocr_result']['height']}\n"
        )
        logger.info(f"{result['description']} OCR Results:\n{ocr_text}")
    
    # Remove axis ticks for cleaner display
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png')
    logger.info("Saved visualization to preprocessing_comparison.png")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='Test enhanced preprocessing for low-quality images')
    parser.add_argument('--image', type=str, default='id_card.jpg',
                       help='Path to the image to process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return
    
    await test_image_processing(args.image)

if __name__ == "__main__":
    asyncio.run(main())

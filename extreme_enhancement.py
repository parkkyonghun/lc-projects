"""
Extreme Image Enhancement for Ultra-Low Quality Images

This module implements the most aggressive enhancement techniques
for images that are extremely poor quality, blurry, noisy, or damaged.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ExtremeImageEnhancer:
    """
    Extreme enhancement for the worst quality images.
    
    Uses the most aggressive techniques available to extract
    any possible text information from severely degraded images.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize extreme enhancer.
        
        Args:
            debug: Enable debug mode to save intermediate steps
        """
        self.debug = debug
        self.debug_counter = 0
    
    def enhance_extreme_quality(self, image: Image.Image) -> List[Image.Image]:
        """
        Apply extreme enhancement and return multiple enhanced versions.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of enhanced images with different processing approaches
        """
        logger.info("Applying extreme enhancement for ultra-low quality image")
        
        # Convert to numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        
        enhanced_versions = []
        
        # Version 1: Maximum contrast and sharpening
        version1 = self._extreme_contrast_enhancement(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version1))
        
        # Version 2: Aggressive denoising with edge preservation
        version2 = self._extreme_denoising(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version2))
        
        # Version 3: Super-resolution with interpolation
        version3 = self._extreme_super_resolution(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version3))
        
        # Version 4: Text-focused enhancement
        version4 = self._extreme_text_enhancement(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version4))
        
        # Version 5: Multi-scale processing
        version5 = self._extreme_multi_scale(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version5))
        
        # Version 6: Frequency domain enhancement
        version6 = self._extreme_frequency_enhancement(img_array.copy())
        enhanced_versions.append(self._array_to_pil(version6))
        
        logger.info(f"Generated {len(enhanced_versions)} extreme enhancement versions")
        return enhanced_versions
    
    def _extreme_contrast_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply maximum contrast enhancement."""
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Extreme CLAHE
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Extreme gamma correction
        gamma = 0.5  # Very aggressive
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                               for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, gamma_table)
        
        # Extreme unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        enhanced = cv2.addWeighted(enhanced, 2.5, gaussian, -1.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Extreme histogram stretching
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        self._save_debug(enhanced, "extreme_contrast")
        return enhanced
    
    def _extreme_denoising(self, img: np.ndarray) -> np.ndarray:
        """Apply extreme denoising while preserving text edges."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Multiple rounds of denoising
        denoised = gray.copy()
        
        # Round 1: Non-local means with strong parameters
        denoised = cv2.fastNlMeansDenoising(denoised, None, 20, 7, 21)
        
        # Round 2: Bilateral filtering
        denoised = cv2.bilateralFilter(denoised, 15, 100, 100)
        
        # Round 3: Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Round 4: Median filtering for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 5)
        
        # Enhance contrast after denoising
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        denoised = clahe.apply(denoised)
        
        self._save_debug(denoised, "extreme_denoising")
        return denoised
    
    def _extreme_super_resolution(self, img: np.ndarray) -> np.ndarray:
        """Apply extreme super-resolution techniques."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Scale up by 4x using multiple methods
        h, w = gray.shape
        new_h, new_w = h * 4, w * 4
        
        # Method 1: Lanczos interpolation
        upscaled1 = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Method 2: Cubic interpolation
        upscaled2 = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Combine methods
        upscaled = cv2.addWeighted(upscaled1, 0.7, upscaled2, 0.3, 0)
        
        # Edge enhancement after upscaling
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        
        # Apply sharpening
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        upscaled = cv2.addWeighted(upscaled, 1.8, gaussian, -0.8, 0)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
        
        self._save_debug(upscaled, "extreme_super_resolution")
        return upscaled
    
    def _extreme_text_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply text-specific extreme enhancement."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Text-specific preprocessing
        enhanced = gray.copy()
        
        # Remove background variations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        background = cv2.medianBlur(background, 21)
        
        # Subtract background
        enhanced = enhanced.astype(np.float32)
        background = background.astype(np.float32)
        enhanced = enhanced - background + 128
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Extreme contrast for text
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(enhanced)
        
        # Text-specific morphological operations
        # Strengthen horizontal text lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_h)
        
        # Strengthen vertical text lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_v)
        
        # Final sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        self._save_debug(enhanced, "extreme_text")
        return enhanced
    
    def _extreme_multi_scale(self, img: np.ndarray) -> np.ndarray:
        """Apply multi-scale enhancement."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Process at multiple scales
        scales = [0.5, 1.0, 2.0]
        enhanced_scales = []
        
        for scale in scales:
            h, w = gray.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize to scale
            if scale != 1.0:
                scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                scaled = gray.copy()
            
            # Apply enhancement at this scale
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced_scale = clahe.apply(scaled)
            
            # Resize back to original size
            if scale != 1.0:
                enhanced_scale = cv2.resize(enhanced_scale, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            enhanced_scales.append(enhanced_scale.astype(np.float32))
        
        # Combine scales
        combined = np.mean(enhanced_scales, axis=0)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Final enhancement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        combined = cv2.filter2D(combined, -1, kernel)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        self._save_debug(combined, "extreme_multi_scale")
        return combined
    
    def _extreme_frequency_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply frequency domain enhancement."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter to enhance edges
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.ones((rows, cols), np.uint8)
        r = 30  # Radius for high-pass filter
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0.3  # Don't completely remove low frequencies
        
        # Apply mask and inverse FFT
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize
        enhanced = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Combine with original
        enhanced = cv2.addWeighted(gray, 0.6, enhanced, 0.4, 0)
        
        self._save_debug(enhanced, "extreme_frequency")
        return enhanced
    
    def _array_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if len(img_array.shape) == 2:
            return Image.fromarray(img_array, mode='L')
        else:
            return Image.fromarray(img_array)
    
    def _save_debug(self, img: np.ndarray, step_name: str):
        """Save debug image if debug mode is enabled."""
        if self.debug:
            filename = f"extreme_debug_{self.debug_counter:02d}_{step_name}.png"
            cv2.imwrite(filename, img)
            self.debug_counter += 1
            logger.info(f"Extreme debug image saved: {filename}")


def enhance_with_multiple_approaches(image: Image.Image, debug: bool = False) -> List[Image.Image]:
    """
    Enhance image using multiple extreme approaches.
    
    Args:
        image: Input PIL Image
        debug: Enable debug mode
        
    Returns:
        List of enhanced images
    """
    enhancer = ExtremeImageEnhancer(debug=debug)
    return enhancer.enhance_extreme_quality(image)


def get_best_enhanced_image(image: Image.Image) -> Image.Image:
    """
    Get the best enhanced version using quality assessment.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Best enhanced PIL Image
    """
    enhanced_versions = enhance_with_multiple_approaches(image, debug=False)
    
    # Simple quality assessment - choose the one with highest contrast
    best_image = enhanced_versions[0]
    best_score = 0
    
    for enhanced in enhanced_versions:
        # Convert to grayscale for assessment
        gray = np.array(enhanced.convert('L'))
        
        # Calculate quality score (contrast + sharpness)
        contrast = np.std(gray)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        score = contrast + (sharpness / 1000)  # Normalize sharpness
        
        if score > best_score:
            best_score = score
            best_image = enhanced
    
    logger.info(f"Selected best enhanced image with quality score: {best_score:.2f}")
    return best_image

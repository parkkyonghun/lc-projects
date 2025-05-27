"""
Advanced Image Enhancement Utilities for OCR Optimization

This module provides comprehensive image preprocessing techniques specifically
optimized for Tesseract OCR, with special consideration for Khmer script.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Advanced image enhancement class for OCR preprocessing.

    Provides a comprehensive pipeline of image enhancement techniques
    specifically optimized for text recognition with Tesseract.
    """

    def __init__(self, target_dpi: int = 300, debug: bool = False):
        """
        Initialize the ImageEnhancer.

        Args:
            target_dpi: Target DPI for optimal OCR (default: 300)
            debug: Enable debug mode to save intermediate images
        """
        self.target_dpi = target_dpi
        self.debug = debug
        self.debug_counter = 0

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Main enhancement pipeline for OCR optimization.

        Args:
            image: Input PIL Image

        Returns:
            Enhanced PIL Image optimized for OCR
        """
        # Convert to OpenCV format
        img_cv = self._pil_to_cv2(image)

        # Apply enhancement pipeline
        img_cv = self._resize_optimal(img_cv, image.info.get('dpi', (72, 72))[0])
        img_cv = self._correct_perspective(img_cv)
        img_cv = self._correct_skew_advanced(img_cv)
        img_cv = self._enhance_lighting(img_cv)
        img_cv = self._reduce_noise_advanced(img_cv)
        img_cv = self._enhance_text_contrast(img_cv)
        img_cv = self._binarize_optimal(img_cv)
        img_cv = self._post_process_text(img_cv)

        # Convert back to PIL
        result = self._cv2_to_pil(img_cv)
        result.info['dpi'] = (self.target_dpi, self.target_dpi)

        return result

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        # Convert to RGB if needed, then to grayscale
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        return np.array(pil_image)

    def _cv2_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        return Image.fromarray(cv_image)

    def _save_debug(self, img: np.ndarray, step_name: str):
        """Save debug image if debug mode is enabled."""
        if self.debug:
            filename = f"debug_{self.debug_counter:02d}_{step_name}.png"
            cv2.imwrite(filename, img)
            self.debug_counter += 1
            logger.info(f"Debug image saved: {filename}")

    def _resize_optimal(self, img: np.ndarray, current_dpi: float) -> np.ndarray:
        """Resize image to optimal DPI for OCR."""
        scale_factor = self.target_dpi / current_dpi if current_dpi != self.target_dpi else 1

        # Ensure minimum size
        h, w = img.shape[:2]
        min_height, min_width = 600, 800

        new_height = max(int(h * scale_factor), min_height)
        new_width = max(int(w * scale_factor), min_width)

        if scale_factor != 1 or new_height != h or new_width != w:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        self._save_debug(img, "resized")
        return img

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct perspective distortion in ID cards."""
        # Find contours to detect the card boundary
        edges = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (likely the card)
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate contour to get corner points
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # If we found a quadrilateral, apply perspective correction
            if len(approx) == 4:
                # Order points: top-left, top-right, bottom-right, bottom-left
                points = self._order_points(approx.reshape(4, 2))

                # Calculate the dimensions of the corrected image
                width = max(
                    np.linalg.norm(points[1] - points[0]),
                    np.linalg.norm(points[2] - points[3])
                )
                height = max(
                    np.linalg.norm(points[3] - points[0]),
                    np.linalg.norm(points[2] - points[1])
                )

                # Define destination points
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)

                # Apply perspective transformation
                matrix = cv2.getPerspectiveTransform(points.astype(np.float32), dst)
                img = cv2.warpPerspective(img, matrix, (int(width), int(height)))

        self._save_debug(img, "perspective_corrected")
        return img

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y-coordinate
        y_sorted = pts[np.argsort(pts[:, 1])]

        # Get top and bottom pairs
        top = y_sorted[:2]
        bottom = y_sorted[2:]

        # Sort top pair by x-coordinate (left to right)
        top = top[np.argsort(top[:, 0])]

        # Sort bottom pair by x-coordinate (right to left for clockwise order)
        bottom = bottom[np.argsort(bottom[:, 0])[::-1]]

        return np.array([top[0], top[1], bottom[0], bottom[1]])

    def _correct_skew_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced skew detection and correction using multiple methods."""
        # Method 1: Hough Line Transform
        skew_angle = self._detect_skew_hough(img)

        # Method 2: Projection Profile (fallback)
        if abs(skew_angle) < 0.1:  # If Hough method didn't find significant skew
            skew_angle = self._detect_skew_projection(img)

        # Apply correction if skew is significant
        if abs(skew_angle) > 0.1:
            img = self._rotate_image(img, skew_angle)
            logger.info(f"Corrected skew angle: {skew_angle:.2f} degrees")

        self._save_debug(img, "skew_corrected")
        return img

    def _detect_skew_hough(self, img: np.ndarray) -> float:
        """Detect skew using Hough Line Transform."""
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -30 <= angle <= 30:  # Only consider reasonable angles
                    angles.append(angle)

            if angles:
                return np.median(angles)

        return 0.0

    def _detect_skew_projection(self, img: np.ndarray) -> float:
        """Detect skew using projection profile method."""
        # This is a simplified version - can be enhanced further
        angles = np.arange(-5, 6, 0.5)  # Test angles from -5 to 5 degrees
        max_variance = 0
        best_angle = 0

        for angle in angles:
            rotated = self._rotate_image(img, angle)
            projection = np.sum(rotated, axis=1)
            variance = np.var(projection)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        return best_angle

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))

        # Adjust translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _enhance_lighting(self, img: np.ndarray) -> np.ndarray:
        """Enhance lighting and remove shadows."""
        # Remove shadows using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        background = cv2.medianBlur(background, 19)

        # Normalize lighting
        img_float = img.astype(np.float32)
        background_float = background.astype(np.float32)
        normalized = img_float / (background_float + 1e-6) * 255
        img = np.clip(normalized, 0, 255).astype(np.uint8)

        self._save_debug(img, "lighting_enhanced")
        return img

    def _reduce_noise_advanced(self, img: np.ndarray) -> np.ndarray:
        """Advanced noise reduction while preserving text."""
        # Bilateral filter for edge-preserving smoothing
        img = cv2.bilateralFilter(img, 9, 75, 75)

        # Non-local means denoising
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        self._save_debug(img, "noise_reduced")
        return img

    def _enhance_text_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance contrast specifically for text regions."""
        # Apply CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Gamma correction for better text visibility
        gamma = 1.2
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, gamma_table)

        self._save_debug(img, "contrast_enhanced")
        return img

    def _binarize_optimal(self, img: np.ndarray) -> np.ndarray:
        """Apply optimal binarization for text recognition."""
        # Multiple binarization methods
        binary1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        _, binary3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine methods for optimal result
        combined = cv2.bitwise_and(binary1, binary2)
        mask = cv2.bitwise_xor(binary1, binary2)
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(binary3, mask))

        self._save_debug(combined, "binarized")
        return combined

    def _post_process_text(self, img: np.ndarray) -> np.ndarray:
        """Post-process binary image for optimal text recognition."""
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Connect broken characters
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

        # Ensure proper polarity (black text on white background)
        if np.sum(img == 0) > np.sum(img == 255):
            img = cv2.bitwise_not(img)

        self._save_debug(img, "post_processed")
        return img


def create_preprocessing_pipeline(target_dpi: int = 300, debug: bool = False) -> ImageEnhancer:
    """
    Factory function to create an optimized preprocessing pipeline.

    Args:
        target_dpi: Target DPI for OCR optimization
        debug: Enable debug mode

    Returns:
        Configured ImageEnhancer instance
    """
    return ImageEnhancer(target_dpi=target_dpi, debug=debug)


def assess_image_quality_comprehensive(img: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive image quality assessment for OCR preprocessing.

    Args:
        img: Input image as numpy array

    Returns:
        Dictionary with detailed quality metrics
    """
    metrics = {}

    # Basic metrics
    metrics['mean_brightness'] = float(np.mean(img))
    metrics['std_contrast'] = float(np.std(img))
    metrics['min_value'] = float(np.min(img))
    metrics['max_value'] = float(np.max(img))

    # Sharpness using Laplacian variance
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    metrics['sharpness'] = float(laplacian.var())

    # Noise estimation
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    noise = cv2.filter2D(img, -1, kernel)
    metrics['noise_level'] = float(np.std(noise))

    # Text-background separation
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_pixels = np.sum(binary == 0)
    background_pixels = np.sum(binary == 255)
    total_pixels = text_pixels + background_pixels

    if total_pixels > 0:
        metrics['text_ratio'] = float(text_pixels / total_pixels)
        metrics['background_ratio'] = float(background_pixels / total_pixels)
    else:
        metrics['text_ratio'] = 0.0
        metrics['background_ratio'] = 0.0

    # Edge density (indicator of text presence)
    edges = cv2.Canny(img, 50, 150)
    metrics['edge_density'] = float(np.sum(edges > 0) / edges.size)

    return metrics

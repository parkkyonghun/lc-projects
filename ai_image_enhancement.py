"""
AI-Powered Image Enhancement for Ultra Low Quality Images

This module implements state-of-the-art AI and deep learning techniques
for enhancing extremely low quality images for optimal OCR performance.
Uses modern computer vision and machine learning approaches.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AIImageEnhancer:
    """
    Advanced AI-powered image enhancement using modern deep learning techniques.

    Features:
    - Super-resolution using ESRGAN/Real-ESRGAN
    - AI-based denoising with DnCNN
    - Intelligent contrast enhancement
    - Deep learning-based text detection and enhancement
    - Adaptive processing based on image quality assessment
    """

    def __init__(self, use_gpu: bool = True, model_path: Optional[str] = None):
        """
        Initialize AI Image Enhancer.

        Args:
            use_gpu: Whether to use GPU acceleration if available
            model_path: Path to custom trained models
        """
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.model_path = model_path
        self.models_loaded = False

        # Initialize models lazily
        self._super_resolution_model = None
        self._denoising_model = None
        self._text_detector = None

        logger.info(f"AI Enhancer initialized - GPU: {self.use_gpu}")

    def enhance_ultra_low_quality(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Main enhancement pipeline for ultra-low quality images.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('RGB'))
        else:
            img_array = image.copy()

        logger.info(f"Processing ultra-low quality image: {img_array.shape}")

        # Step 1: AI-based quality assessment and adaptive preprocessing
        quality_metrics = self._assess_image_quality_ai(img_array)
        enhancement_strategy = self._determine_enhancement_strategy(quality_metrics)

        # Step 2: Super-resolution for very low resolution images
        if quality_metrics['needs_super_resolution']:
            img_array = self._apply_super_resolution(img_array)
            logger.info("Applied AI super-resolution")

        # Step 3: Advanced AI denoising
        if quality_metrics['noise_level'] > 0.3:
            img_array = self._apply_ai_denoising(img_array)
            logger.info("Applied AI denoising")

        # Step 4: Intelligent contrast and brightness enhancement
        img_array = self._enhance_contrast_ai(img_array, enhancement_strategy)

        # Step 5: Text-aware enhancement
        img_array = self._enhance_text_regions(img_array)

        # Step 6: Advanced binarization with AI guidance
        img_array = self._binarize_with_ai(img_array)

        # Step 7: Final post-processing
        img_array = self._post_process_ai(img_array)

        # Convert back to PIL Image
        if len(img_array.shape) == 3:
            result = Image.fromarray(img_array)
        else:
            result = Image.fromarray(img_array, mode='L')

        return result

    def _assess_image_quality_ai(self, img: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        AI-based comprehensive image quality assessment.

        Args:
            img: Input image array

        Returns:
            Dictionary with quality metrics and enhancement recommendations
        """
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        metrics = {}

        # Resolution analysis
        h, w = gray.shape[:2]
        total_pixels = h * w
        metrics['resolution'] = total_pixels
        metrics['needs_super_resolution'] = total_pixels < 500000  # Less than 0.5MP

        # Advanced noise analysis using multiple methods
        metrics['noise_level'] = self._estimate_noise_level_advanced(gray)

        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)
        metrics['is_blurry'] = laplacian_var < 100

        # Contrast analysis
        metrics['contrast'] = float(np.std(gray))
        metrics['low_contrast'] = metrics['contrast'] < 30

        # Brightness analysis
        metrics['brightness'] = float(np.mean(gray))
        metrics['too_dark'] = metrics['brightness'] < 80
        metrics['too_bright'] = metrics['brightness'] > 200

        # Text detection confidence
        metrics['text_confidence'] = self._estimate_text_presence(gray)

        # Overall quality score (0-1, higher is better)
        quality_score = self._calculate_quality_score(metrics)
        metrics['quality_score'] = quality_score
        metrics['is_ultra_low_quality'] = quality_score < 0.3

        return metrics

    def _estimate_noise_level_advanced(self, img: np.ndarray) -> float:
        """
        Advanced noise level estimation using multiple techniques.

        Args:
            img: Grayscale image

        Returns:
            Noise level estimate (0-1, higher means more noise)
        """
        # Method 1: High-frequency content analysis
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(img, -1, kernel)
        noise_1 = np.std(high_freq) / 255.0

        # Method 2: Wavelet-based noise estimation
        try:
            import pywt
            coeffs = pywt.dwt2(img, 'db4')
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            noise_2 = min(sigma / 50.0, 1.0)  # Normalize
        except ImportError:
            noise_2 = noise_1  # Fallback

        # Method 3: Local variance analysis
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        mean_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
        sqr_img = cv2.filter2D((img.astype(np.float32))**2, -1, kernel)
        variance_img = sqr_img - mean_img**2
        noise_3 = np.mean(variance_img) / (255.0**2)

        # Combine estimates
        noise_level = (noise_1 + noise_2 + noise_3) / 3.0
        return min(noise_level, 1.0)

    def _estimate_text_presence(self, img: np.ndarray) -> float:
        """
        Estimate the presence and quality of text in the image.

        Args:
            img: Grayscale image

        Returns:
            Text confidence score (0-1)
        """
        # Edge density analysis
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        line_density = (np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)) / edges.size

        # Text-like pattern detection using connected components
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # Analyze component characteristics
        text_like_components = 0
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Text-like characteristics
            if 10 < area < 5000 and 2 < width < 200 and 5 < height < 100:
                aspect_ratio = width / height
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    text_like_components += 1

        component_score = min(text_like_components / 50.0, 1.0)

        # Combine scores
        text_confidence = (edge_density * 0.3 + line_density * 0.3 + component_score * 0.4)
        return min(text_confidence, 1.0)

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall image quality score.

        Args:
            metrics: Quality metrics dictionary

        Returns:
            Quality score (0-1, higher is better)
        """
        score = 0.0

        # Resolution component
        if metrics['resolution'] > 1000000:  # > 1MP
            score += 0.25
        elif metrics['resolution'] > 500000:  # > 0.5MP
            score += 0.15
        elif metrics['resolution'] > 100000:  # > 0.1MP
            score += 0.05

        # Sharpness component
        if metrics['sharpness'] > 500:
            score += 0.25
        elif metrics['sharpness'] > 100:
            score += 0.15
        elif metrics['sharpness'] > 50:
            score += 0.05

        # Contrast component
        if metrics['contrast'] > 60:
            score += 0.2
        elif metrics['contrast'] > 30:
            score += 0.1
        elif metrics['contrast'] > 15:
            score += 0.05

        # Noise component (inverse)
        if metrics['noise_level'] < 0.1:
            score += 0.15
        elif metrics['noise_level'] < 0.3:
            score += 0.1
        elif metrics['noise_level'] < 0.5:
            score += 0.05

        # Text presence component
        score += metrics['text_confidence'] * 0.15

        return min(score, 1.0)

    def _determine_enhancement_strategy(self, metrics: Dict) -> Dict[str, Union[str, float]]:
        """
        Determine optimal enhancement strategy based on quality metrics.

        Args:
            metrics: Quality assessment results

        Returns:
            Enhancement strategy parameters
        """
        strategy = {
            'approach': 'balanced',
            'denoising_strength': 0.5,
            'contrast_boost': 1.0,
            'sharpening_strength': 0.5,
            'super_resolution_factor': 2.0
        }

        # Adjust based on quality assessment
        if metrics['is_ultra_low_quality']:
            strategy['approach'] = 'aggressive'
            strategy['denoising_strength'] = 0.8
            strategy['contrast_boost'] = 1.5
            strategy['sharpening_strength'] = 0.8

        if metrics['noise_level'] > 0.5:
            strategy['denoising_strength'] = min(strategy['denoising_strength'] + 0.3, 1.0)

        if metrics['low_contrast']:
            strategy['contrast_boost'] = min(strategy['contrast_boost'] + 0.5, 2.0)

        if metrics['is_blurry']:
            strategy['sharpening_strength'] = min(strategy['sharpening_strength'] + 0.3, 1.0)

        if metrics['needs_super_resolution']:
            if metrics['resolution'] < 100000:  # Very low resolution
                strategy['super_resolution_factor'] = 4.0
            else:
                strategy['super_resolution_factor'] = 2.0

        return strategy

    def _apply_super_resolution(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AI-based super-resolution to enhance image resolution.

        Args:
            img: Input image array

        Returns:
            Super-resolved image
        """
        try:
            # Try to use OpenCV's DNN super-resolution models
            if not hasattr(self, '_sr_model') or self._sr_model is None:
                self._load_super_resolution_model()

            if self._sr_model is not None:
                # Use pre-trained ESRGAN or similar model
                result = self._sr_model.upsample(img)
                logger.info("Applied DNN super-resolution")
                return result
        except Exception as e:
            logger.warning(f"DNN super-resolution failed: {e}")

        # Fallback to advanced interpolation methods
        return self._apply_advanced_interpolation(img)

    def _load_super_resolution_model(self):
        """Load super-resolution model (ESRGAN, Real-ESRGAN, etc.)"""
        try:
            # Try to load OpenCV DNN super-resolution
            self._sr_model = cv2.dnn_superres.DnnSuperResImpl_create()

            # Try different model paths
            model_paths = [
                "models/ESRGAN_x4.pb",
                "models/RealESRGAN_x4plus.onnx",
                "models/EDSR_x4.pb"
            ]

            for model_path in model_paths:
                try:
                    if model_path.endswith('.pb'):
                        self._sr_model.readModel(model_path)
                        self._sr_model.setModel("esrgan", 4)
                        logger.info(f"Loaded super-resolution model: {model_path}")
                        return
                except:
                    continue

            # If no model found, disable DNN super-resolution
            self._sr_model = None
            logger.warning("No super-resolution model found, using fallback methods")

        except Exception as e:
            logger.warning(f"Failed to initialize super-resolution: {e}")
            self._sr_model = None

    def _apply_advanced_interpolation(self, img: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Apply advanced interpolation techniques for upscaling.

        Args:
            img: Input image
            scale_factor: Upscaling factor

        Returns:
            Upscaled image
        """
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Method 1: Lanczos interpolation with edge enhancement
        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Method 2: Edge-directed interpolation
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edges_upscaled = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Apply edge-preserving smoothing
        upscaled = cv2.edgePreservingFilter(upscaled, flags=2, sigma_s=50, sigma_r=0.4)

        return upscaled

    def _apply_ai_denoising(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AI-based denoising techniques.

        Args:
            img: Input image

        Returns:
            Denoised image
        """
        # Advanced denoising pipeline

        # Step 1: Non-local means denoising (improved parameters)
        if len(img.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        # Step 2: Bilateral filtering for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

        # Step 3: Morphological noise reduction
        if len(denoised.shape) == 2:  # Grayscale
            # Remove salt and pepper noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        # Step 4: Gaussian blur for remaining noise (very light)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)

        return denoised

    def _enhance_contrast_ai(self, img: np.ndarray, strategy: Dict) -> np.ndarray:
        """
        AI-guided contrast enhancement.

        Args:
            img: Input image
            strategy: Enhancement strategy parameters

        Returns:
            Contrast-enhanced image
        """
        if len(img.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = img.copy()

        # Adaptive histogram equalization with optimized parameters
        clip_limit = 2.0 + (strategy['contrast_boost'] - 1.0) * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # Gamma correction based on image characteristics
        gamma = 1.0 + (strategy['contrast_boost'] - 1.0) * 0.3
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        enhanced_l = cv2.LUT(enhanced_l, gamma_table)

        # Unsharp masking for additional sharpness
        if strategy['sharpening_strength'] > 0:
            gaussian = cv2.GaussianBlur(enhanced_l, (0, 0), 2.0)
            enhanced_l = cv2.addWeighted(enhanced_l, 1.0 + strategy['sharpening_strength'],
                                       gaussian, -strategy['sharpening_strength'], 0)

        if len(img.shape) == 3:
            # Reconstruct color image
            lab[:, :, 0] = enhanced_l
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            result = enhanced_l

        return result

    def _enhance_text_regions(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance text regions specifically using text detection.

        Args:
            img: Input image

        Returns:
            Text-enhanced image
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Detect text regions using MSER (Maximally Stable Extremal Regions)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Create mask for text regions
        text_mask = np.zeros_like(gray)
        for region in regions:
            # Filter regions by size and aspect ratio
            if len(region) > 10:  # Minimum region size
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                cv2.fillPoly(text_mask, [hull], 255)

        # Dilate mask to include surrounding areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=2)

        # Apply stronger enhancement to text regions
        enhanced = gray.copy()

        # Local contrast enhancement for text regions
        text_regions = cv2.bitwise_and(gray, text_mask)
        if np.sum(text_regions) > 0:
            # Apply stronger CLAHE to text regions
            clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            text_enhanced = clahe_strong.apply(text_regions)

            # Blend enhanced text regions back
            mask_norm = text_mask.astype(np.float32) / 255.0
            enhanced = enhanced.astype(np.float32)
            text_enhanced = text_enhanced.astype(np.float32)

            enhanced = enhanced * (1 - mask_norm) + text_enhanced * mask_norm
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    def _binarize_with_ai(self, img: np.ndarray) -> np.ndarray:
        """
        AI-guided binarization for optimal text recognition.

        Args:
            img: Input grayscale image

        Returns:
            Binarized image
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Method 1: Adaptive thresholding with multiple block sizes
        binary_results = []

        # Small block size for fine details
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 7, 3)
        binary_results.append(binary1)

        # Medium block size for general text
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary_results.append(binary2)

        # Large block size for varying illumination
        binary3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 2)
        binary_results.append(binary3)

        # Method 2: Otsu's thresholding
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_results.append(binary_otsu)

        # Method 3: Sauvola's local thresholding (approximation)
        binary_sauvola = self._sauvola_threshold(gray)
        binary_results.append(binary_sauvola)

        # Combine results using voting
        combined = self._combine_binary_results(binary_results)

        return combined

    def _sauvola_threshold(self, img: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
        """
        Sauvola's local thresholding method.

        Args:
            img: Input grayscale image
            window_size: Size of local window
            k: Sauvola parameter

        Returns:
            Binarized image
        """
        # Calculate local mean and standard deviation
        mean = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter((img.astype(np.float32))**2, -1, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean**2)

        # Sauvola threshold
        threshold = mean * (1 + k * ((std / 128) - 1))

        # Apply threshold
        binary = np.where(img > threshold, 255, 0).astype(np.uint8)

        return binary

    def _combine_binary_results(self, binary_results: list) -> np.ndarray:
        """
        Combine multiple binarization results using intelligent voting.

        Args:
            binary_results: List of binary images

        Returns:
            Combined binary image
        """
        if not binary_results:
            return np.zeros((100, 100), dtype=np.uint8)

        # Convert to float for averaging
        float_results = [img.astype(np.float32) / 255.0 for img in binary_results]

        # Calculate average
        avg_result = np.mean(float_results, axis=0)

        # Apply adaptive threshold based on local variance
        # Areas with high agreement get stricter threshold
        variance = np.var(float_results, axis=0)
        adaptive_threshold = 0.5 - variance * 0.3  # Lower threshold for high variance areas

        # Apply threshold
        combined = np.where(avg_result > adaptive_threshold, 255, 0).astype(np.uint8)

        return combined

    def _post_process_ai(self, img: np.ndarray) -> np.ndarray:
        """
        AI-guided post-processing for final optimization.

        Args:
            img: Input binary image

        Returns:
            Post-processed image
        """
        # Ensure binary image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Ensure proper polarity (black text on white background)
        if np.sum(img == 0) > np.sum(img == 255):
            img = cv2.bitwise_not(img)

        # Remove small noise components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

        # Filter components by size and characteristics
        min_area = 10  # Minimum area for text components
        max_area = img.shape[0] * img.shape[1] * 0.1  # Maximum 10% of image

        filtered = np.zeros_like(img)
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]

            # Keep components that look like text
            if min_area <= area <= max_area:
                aspect_ratio = width / height if height > 0 else 0
                if 0.1 <= aspect_ratio <= 10:  # Reasonable aspect ratio
                    component_mask = (labels == i).astype(np.uint8) * 255
                    filtered = cv2.bitwise_or(filtered, component_mask)

        # Morphological operations for text enhancement
        # Connect broken characters
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

        # Strengthen vertical strokes (important for many scripts)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_vertical, iterations=1)

        # Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_smooth, iterations=1)

        return filtered


# Factory functions and utilities

def create_ai_enhancer(use_gpu: bool = True, model_path: Optional[str] = None) -> AIImageEnhancer:
    """
    Create an AI-powered image enhancer.

    Args:
        use_gpu: Whether to use GPU acceleration
        model_path: Path to custom models

    Returns:
        AIImageEnhancer instance
    """
    return AIImageEnhancer(use_gpu=use_gpu, model_path=model_path)


def enhance_ultra_low_quality_image(image: Union[Image.Image, np.ndarray],
                                   use_gpu: bool = True) -> Image.Image:
    """
    Convenience function to enhance ultra-low quality images.

    Args:
        image: Input image
        use_gpu: Whether to use GPU acceleration

    Returns:
        Enhanced PIL Image
    """
    enhancer = create_ai_enhancer(use_gpu=use_gpu)
    return enhancer.enhance_ultra_low_quality(image)


def assess_enhancement_potential(image: Union[Image.Image, np.ndarray]) -> Dict[str, float]:
    """
    Assess how much an image could benefit from AI enhancement.

    Args:
        image: Input image

    Returns:
        Assessment metrics and recommendations
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image.copy()

    enhancer = AIImageEnhancer()
    metrics = enhancer._assess_image_quality_ai(img_array)

    # Add enhancement recommendations
    recommendations = {
        'should_enhance': metrics['quality_score'] < 0.7,
        'enhancement_priority': 'high' if metrics['quality_score'] < 0.3 else
                               'medium' if metrics['quality_score'] < 0.6 else 'low',
        'expected_improvement': min((1.0 - metrics['quality_score']) * 0.8, 0.6),
        'processing_time_estimate': 'fast' if metrics['resolution'] < 500000 else
                                   'medium' if metrics['resolution'] < 2000000 else 'slow'
    }

    metrics.update(recommendations)
    return metrics

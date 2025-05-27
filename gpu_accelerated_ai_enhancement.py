"""
GPU-Accelerated AI Image Enhancement for OCR

This module provides GPU-accelerated image enhancement specifically optimized
for OCR preprocessing, with automatic fallback to CPU when GPU is unavailable.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import logging
from typing import Optional, Union, Dict, Any
import time
from pathlib import Path

# Import existing AI enhancement
from ai_image_enhancement import AIImageEnhancer
from ai_enhancement_config import AIEnhancementConfig

logger = logging.getLogger(__name__)


class GPUAcceleratedAIEnhancer:
    """
    GPU-accelerated AI image enhancer with automatic device selection and optimization.
    
    Features:
    - Automatic GPU/CPU device selection
    - GPU memory management and optimization
    - Performance monitoring and benchmarking
    - Fallback to CPU when GPU unavailable
    - Batch processing for multiple images
    """
    
    def __init__(self, config: Optional[AIEnhancementConfig] = None, force_cpu: bool = False):
        """
        Initialize GPU-accelerated AI enhancer.
        
        Args:
            config: AI enhancement configuration
            force_cpu: Force CPU usage even if GPU available
        """
        self.config = config or AIEnhancementConfig()
        self.force_cpu = force_cpu
        
        # Device selection and setup
        self.device = self._setup_device()
        self.gpu_available = self.device.type == 'cuda'
        
        # Performance tracking
        self.performance_stats = {
            'total_images_processed': 0,
            'gpu_processing_time': 0.0,
            'cpu_processing_time': 0.0,
            'gpu_memory_usage': 0.0,
            'speedup_factor': 1.0
        }
        
        # Initialize models
        self.models_loaded = False
        self._super_resolution_model = None
        self._denoising_model = None
        
        logger.info(f"GPU-Accelerated AI Enhancer initialized - Device: {self.device}")
        if self.gpu_available:
            self._log_gpu_info()
    
    def _setup_device(self) -> torch.device:
        """Setup and configure the optimal device for processing."""
        if self.force_cpu:
            logger.info("Forced CPU usage")
            return torch.device('cpu')
        
        if not self.config.use_gpu:
            logger.info("GPU disabled in configuration")
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Configure GPU optimizations
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info(f"GPU device selected: {device}")
            return device
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device('cpu')
    
    def _log_gpu_info(self):
        """Log GPU information and capabilities."""
        if not self.gpu_available:
            return
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check OpenCV CUDA support
            opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount()
            logger.info(f"OpenCV CUDA devices: {opencv_cuda}")
            
        except Exception as e:
            logger.warning(f"Could not retrieve GPU info: {e}")
    
    def enhance_image_gpu(self, image: Union[Image.Image, np.ndarray], 
                         enhancement_type: str = "ultra_low_quality") -> Image.Image:
        """
        Enhance image using GPU acceleration.
        
        Args:
            image: Input image (PIL Image or numpy array)
            enhancement_type: Type of enhancement to apply
            
        Returns:
            Enhanced PIL Image
        """
        start_time = time.time()
        
        try:
            # Convert to tensor and move to device
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Ensure proper format for GPU processing
            if len(image_array.shape) == 3:
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image_array).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
            
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Apply GPU-accelerated enhancements
            enhanced_tensor = self._apply_gpu_enhancements(image_tensor, enhancement_type)
            
            # Convert back to PIL Image
            enhanced_array = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_array = (enhanced_array * 255).astype(np.uint8)
            enhanced_image = Image.fromarray(enhanced_array)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['total_images_processed'] += 1
            self.performance_stats['gpu_processing_time'] += processing_time
            
            # Monitor GPU memory usage
            if self.gpu_available:
                memory_used = torch.cuda.memory_allocated() / 1024**3
                self.performance_stats['gpu_memory_usage'] = max(
                    self.performance_stats['gpu_memory_usage'], memory_used
                )
            
            logger.info(f"GPU enhancement completed in {processing_time:.3f}s")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"GPU enhancement failed: {e}")
            # Fallback to CPU processing
            return self._fallback_cpu_enhancement(image, enhancement_type)
    
    def _apply_gpu_enhancements(self, image_tensor: torch.Tensor, 
                               enhancement_type: str) -> torch.Tensor:
        """Apply GPU-accelerated image enhancements."""
        enhanced = image_tensor
        
        if enhancement_type in ["ultra_low_quality", "low_quality"]:
            # Super-resolution
            if self.config.enable_super_resolution:
                enhanced = self._gpu_super_resolution(enhanced)
            
            # Denoising
            if self.config.enable_ai_denoising:
                enhanced = self._gpu_denoising(enhanced)
            
            # Contrast enhancement
            enhanced = self._gpu_contrast_enhancement(enhanced)
            
            # Sharpening
            enhanced = self._gpu_sharpening(enhanced)
        
        return enhanced
    
    def _gpu_super_resolution(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply GPU-accelerated super-resolution."""
        try:
            # Simple bicubic upscaling on GPU (placeholder for actual SR model)
            scale_factor = self.config.sr_scale_factor
            upscaled = torch.nn.functional.interpolate(
                image_tensor, 
                scale_factor=scale_factor, 
                mode='bicubic', 
                align_corners=False
            )
            return upscaled
        except Exception as e:
            logger.warning(f"GPU super-resolution failed: {e}")
            return image_tensor
    
    def _gpu_denoising(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply GPU-accelerated denoising."""
        try:
            # Simple Gaussian blur denoising on GPU
            kernel_size = 3
            sigma = self.config.denoising_strength
            
            # Create Gaussian kernel
            kernel = self._create_gaussian_kernel(kernel_size, sigma).to(self.device)
            
            # Apply convolution
            if len(image_tensor.shape) == 4:  # Batch processing
                channels = image_tensor.shape[1]
                kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
                denoised = torch.nn.functional.conv2d(
                    image_tensor, kernel, padding=kernel_size//2, groups=channels
                )
            else:
                denoised = image_tensor
            
            return denoised
        except Exception as e:
            logger.warning(f"GPU denoising failed: {e}")
            return image_tensor
    
    def _gpu_contrast_enhancement(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply GPU-accelerated contrast enhancement."""
        try:
            # Histogram equalization approximation
            enhanced = torch.clamp(
                (image_tensor - 0.5) * self.config.contrast_boost_factor + 0.5,
                0.0, 1.0
            )
            return enhanced
        except Exception as e:
            logger.warning(f"GPU contrast enhancement failed: {e}")
            return image_tensor
    
    def _gpu_sharpening(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply GPU-accelerated sharpening."""
        try:
            # Unsharp masking on GPU
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=torch.float32).to(self.device)
            
            if len(image_tensor.shape) == 4:
                channels = image_tensor.shape[1]
                kernel = kernel.expand(channels, 1, 3, 3)
                sharpened = torch.nn.functional.conv2d(
                    image_tensor, kernel, padding=1, groups=channels
                )
            else:
                sharpened = image_tensor
            
            return torch.clamp(sharpened, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"GPU sharpening failed: {e}")
            return image_tensor
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a Gaussian kernel for GPU processing."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    
    def _fallback_cpu_enhancement(self, image: Union[Image.Image, np.ndarray], 
                                 enhancement_type: str) -> Image.Image:
        """Fallback to CPU-based enhancement."""
        start_time = time.time()
        
        try:
            # Use existing CPU-based AI enhancer
            cpu_enhancer = AIImageEnhancer(use_gpu=False)
            enhanced_image = cpu_enhancer.enhance_ultra_low_quality(image)
            
            processing_time = time.time() - start_time
            self.performance_stats['cpu_processing_time'] += processing_time
            
            logger.info(f"CPU fallback enhancement completed in {processing_time:.3f}s")
            return enhanced_image
            
        except Exception as e:
            logger.error(f"CPU fallback enhancement failed: {e}")
            # Return original image if all else fails
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
            return image
    
    def batch_enhance_images(self, images: list, enhancement_type: str = "ultra_low_quality") -> list:
        """
        Enhance multiple images in batch for better GPU utilization.
        
        Args:
            images: List of images to enhance
            enhancement_type: Type of enhancement to apply
            
        Returns:
            List of enhanced images
        """
        if not self.gpu_available or len(images) == 1:
            # Process individually if no GPU or single image
            return [self.enhance_image_gpu(img, enhancement_type) for img in images]
        
        logger.info(f"Batch processing {len(images)} images on GPU")
        
        try:
            # Convert all images to tensors
            image_tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                else:
                    img_array = img
                
                if len(img_array.shape) == 3:
                    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                else:
                    tensor = torch.from_numpy(img_array).float() / 255.0
                    tensor = tensor.unsqueeze(0)
                
                image_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(image_tensors).to(self.device)
            
            # Process batch
            enhanced_batch = self._apply_gpu_enhancements(batch_tensor, enhancement_type)
            
            # Convert back to PIL Images
            enhanced_images = []
            for i in range(enhanced_batch.shape[0]):
                enhanced_array = enhanced_batch[i].permute(1, 2, 0).cpu().numpy()
                enhanced_array = (enhanced_array * 255).astype(np.uint8)
                enhanced_images.append(Image.fromarray(enhanced_array))
            
            return enhanced_images
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.enhance_image_gpu(img, enhancement_type) for img in images]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['cpu_processing_time'] > 0 and stats['gpu_processing_time'] > 0:
            avg_gpu_time = stats['gpu_processing_time'] / max(1, stats['total_images_processed'])
            avg_cpu_time = stats['cpu_processing_time'] / max(1, stats['total_images_processed'])
            stats['speedup_factor'] = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1.0
        
        stats['device'] = str(self.device)
        stats['gpu_available'] = self.gpu_available
        
        return stats
    
    def cleanup(self):
        """Cleanup GPU resources."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")


# Factory functions
def create_gpu_ai_enhancer(config: Optional[AIEnhancementConfig] = None, 
                          force_cpu: bool = False) -> GPUAcceleratedAIEnhancer:
    """Create a GPU-accelerated AI enhancer."""
    return GPUAcceleratedAIEnhancer(config=config, force_cpu=force_cpu)


def enhance_image_with_gpu(image: Union[Image.Image, np.ndarray], 
                          enhancement_type: str = "ultra_low_quality") -> Image.Image:
    """Convenience function for GPU-accelerated image enhancement."""
    enhancer = create_gpu_ai_enhancer()
    return enhancer.enhance_image_gpu(image, enhancement_type)

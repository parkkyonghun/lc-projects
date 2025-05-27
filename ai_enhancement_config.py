"""
Configuration for AI-powered image enhancement system.

This module provides configuration classes and presets for the AI enhancement
pipeline, including settings for different quality levels and use cases.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import os


@dataclass
class AIEnhancementConfig:
    """Configuration for AI-powered image enhancement."""

    # General settings
    use_gpu: bool = True
    target_dpi: int = 300
    debug_mode: bool = False
    save_intermediate: bool = False

    # Quality assessment thresholds
    ultra_low_quality_threshold: float = 0.3
    low_quality_threshold: float = 0.6
    noise_threshold: float = 0.3
    blur_threshold: float = 100.0

    # Super-resolution settings
    enable_super_resolution: bool = True
    sr_scale_factor: float = 2.0
    sr_model_path: Optional[str] = None
    min_resolution_for_sr: int = 100000  # Minimum pixels to trigger SR

    # Denoising settings
    enable_ai_denoising: bool = True
    denoising_strength: float = 0.7
    preserve_edges: bool = True

    # Contrast enhancement
    adaptive_contrast: bool = True
    contrast_boost_factor: float = 1.2
    gamma_correction: float = 1.1

    # Text enhancement
    enable_text_detection: bool = True
    text_enhancement_strength: float = 1.5
    mser_delta: int = 5
    mser_min_area: int = 60
    mser_max_area: int = 14400

    # Binarization settings
    multi_method_binarization: bool = True
    sauvola_k: float = 0.2
    sauvola_window_size: int = 15
    adaptive_block_sizes: List[int] = None

    # Post-processing
    remove_small_components: bool = True
    min_component_area: int = 10
    max_component_area_ratio: float = 0.1
    connect_broken_chars: bool = True

    # Performance settings
    max_image_size: int = 4000000  # Maximum pixels to process
    processing_timeout: int = 300  # Seconds
    memory_limit_mb: int = 2048

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.adaptive_block_sizes is None:
            self.adaptive_block_sizes = [7, 11, 15, 21]


@dataclass
class ModelConfig:
    """Configuration for AI models."""

    # Super-resolution models
    sr_models: Dict[str, str] = None

    # Denoising models
    denoising_models: Dict[str, str] = None

    # Text detection models
    text_detection_models: Dict[str, str] = None

    # Model download URLs
    model_urls: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default model configurations."""
        if self.sr_models is None:
            self.sr_models = {
                'esrgan_x4': 'models/ESRGAN_x4.pb',
                'real_esrgan_x4': 'models/RealESRGAN_x4plus.onnx',
                'edsr_x4': 'models/EDSR_x4.pb',
                'srcnn_x2': 'models/SRCNN_x2.pb'
            }

        if self.denoising_models is None:
            self.denoising_models = {
                'dncnn': 'models/DnCNN.onnx',
                'ffdnet': 'models/FFDNet.onnx'
            }

        if self.text_detection_models is None:
            self.text_detection_models = {
                'east': 'models/frozen_east_text_detection.pb',
                'craft': 'models/craft_mlt_25k.pth'
            }

        if self.model_urls is None:
            self.model_urls = {
                'esrgan_x4': 'https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/models/ESRGAN_x4.pb',
                'edsr_x4': 'https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/models/EDSR_x4.pb',
                'east': 'https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/frozen_east_text_detection.pb'
            }


# Predefined configurations for different scenarios

ULTRA_LOW_QUALITY_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    ultra_low_quality_threshold=0.2,
    enable_super_resolution=True,
    sr_scale_factor=4.0,
    denoising_strength=0.9,
    contrast_boost_factor=1.8,
    gamma_correction=1.3,
    text_enhancement_strength=2.0,
    adaptive_block_sizes=[5, 7, 9, 11, 15],
    connect_broken_chars=True,
    debug_mode=False
)

LOW_QUALITY_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    ultra_low_quality_threshold=0.3,
    low_quality_threshold=0.6,
    enable_super_resolution=True,
    sr_scale_factor=2.0,
    denoising_strength=0.7,
    contrast_boost_factor=1.4,
    gamma_correction=1.2,
    text_enhancement_strength=1.5,
    adaptive_block_sizes=[7, 11, 15],
    debug_mode=False
)

MEDIUM_QUALITY_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    ultra_low_quality_threshold=0.4,
    low_quality_threshold=0.7,
    enable_super_resolution=False,  # Not needed for medium quality
    denoising_strength=0.5,
    contrast_boost_factor=1.2,
    gamma_correction=1.1,
    text_enhancement_strength=1.2,
    adaptive_block_sizes=[9, 13, 17],
    debug_mode=False
)

HIGH_PERFORMANCE_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    enable_super_resolution=False,  # Disabled for speed
    enable_ai_denoising=True,
    denoising_strength=0.6,
    multi_method_binarization=False,  # Use single method for speed
    adaptive_block_sizes=[11],  # Single block size
    remove_small_components=False,  # Skip for speed
    processing_timeout=60,  # Shorter timeout
    debug_mode=False
)

KHMER_OPTIMIZED_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    target_dpi=400,  # Higher DPI for complex scripts
    enable_super_resolution=True,
    sr_scale_factor=2.0,
    denoising_strength=0.6,
    contrast_boost_factor=1.3,
    gamma_correction=1.15,
    text_enhancement_strength=1.8,  # Higher for complex scripts
    adaptive_block_sizes=[7, 11, 15, 19],  # More variety for complex text
    sauvola_k=0.15,  # Adjusted for Khmer characteristics
    sauvola_window_size=17,
    connect_broken_chars=True,  # Important for complex scripts
    min_component_area=8,  # Smaller for fine details
    debug_mode=False
)

# NEW: AI Training-Optimized Configurations
AI_TRAINING_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    target_dpi=600,  # Ultra-high DPI for training data
    enable_super_resolution=True,
    sr_scale_factor=4.0,  # Maximum upscaling for training
    denoising_strength=0.8,
    contrast_boost_factor=1.5,
    gamma_correction=1.2,
    text_enhancement_strength=2.0,
    adaptive_block_sizes=[5, 7, 9, 11, 15, 19, 23],  # More variety
    sauvola_k=0.12,  # Optimized for training data quality
    sauvola_window_size=19,
    connect_broken_chars=True,
    min_component_area=6,  # Capture fine details for training
    save_intermediate=True,  # Save all steps for analysis
    debug_mode=True  # Full debugging for training
)

ACTIVE_LEARNING_CONFIG = AIEnhancementConfig(
    use_gpu=True,
    target_dpi=500,
    enable_super_resolution=True,
    sr_scale_factor=3.0,
    denoising_strength=0.9,  # Aggressive denoising for difficult cases
    contrast_boost_factor=2.0,  # High contrast for edge cases
    gamma_correction=1.4,
    text_enhancement_strength=2.5,  # Maximum text enhancement
    adaptive_block_sizes=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21],  # Full range
    sauvola_k=0.1,  # Aggressive binarization
    sauvola_window_size=21,
    connect_broken_chars=True,
    min_component_area=4,  # Capture smallest details
    save_intermediate=True,
    debug_mode=True
)

DEBUG_CONFIG = AIEnhancementConfig(
    use_gpu=False,  # CPU for debugging
    debug_mode=True,
    save_intermediate=True,
    processing_timeout=600,  # Longer timeout for debugging
    enable_super_resolution=True,
    enable_ai_denoising=True,
    multi_method_binarization=True
)


def get_config_by_name(config_name: str) -> AIEnhancementConfig:
    """
    Get predefined AI enhancement configuration by name.

    Args:
        config_name: Name of the configuration

    Returns:
        AIEnhancementConfig instance

    Raises:
        ValueError: If config_name is not found
    """
    configs = {
        'ultra_low_quality': ULTRA_LOW_QUALITY_CONFIG,
        'low_quality': LOW_QUALITY_CONFIG,
        'medium_quality': MEDIUM_QUALITY_CONFIG,
        'high_performance': HIGH_PERFORMANCE_CONFIG,
        'khmer_optimized': KHMER_OPTIMIZED_CONFIG,
        'ai_training': AI_TRAINING_CONFIG,
        'active_learning': ACTIVE_LEARNING_CONFIG,
        'debug': DEBUG_CONFIG,
        'default': LOW_QUALITY_CONFIG
    }

    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return configs[config_name]


def create_custom_ai_config(**kwargs) -> AIEnhancementConfig:
    """
    Create a custom AI enhancement configuration.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        Custom AIEnhancementConfig
    """
    # Start with default config
    config = AIEnhancementConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return config


def auto_select_config(image_quality_score: float,
                      image_size: Tuple[int, int],
                      processing_priority: str = 'quality') -> AIEnhancementConfig:
    """
    Automatically select the best configuration based on image characteristics.

    Args:
        image_quality_score: Quality score from 0-1
        image_size: (width, height) of the image
        processing_priority: 'quality', 'speed', or 'balanced'

    Returns:
        Optimal AIEnhancementConfig
    """
    total_pixels = image_size[0] * image_size[1]

    # Select base config based on quality
    if image_quality_score < 0.3:
        base_config = ULTRA_LOW_QUALITY_CONFIG
    elif image_quality_score < 0.6:
        base_config = LOW_QUALITY_CONFIG
    else:
        base_config = MEDIUM_QUALITY_CONFIG

    # Adjust based on processing priority
    if processing_priority == 'speed':
        # Optimize for speed
        config = create_custom_ai_config(
            **base_config.__dict__,
            enable_super_resolution=False,
            multi_method_binarization=False,
            adaptive_block_sizes=[11],
            remove_small_components=False
        )
    elif processing_priority == 'quality':
        # Optimize for quality
        config = create_custom_ai_config(
            **base_config.__dict__,
            enable_super_resolution=total_pixels < 1000000,
            multi_method_binarization=True,
            save_intermediate=True
        )
    else:  # balanced
        config = base_config

    # Adjust for image size
    if total_pixels > 4000000:  # Very large image
        config.enable_super_resolution = False
        config.processing_timeout = 600
    elif total_pixels < 100000:  # Very small image
        config.enable_super_resolution = True
        config.sr_scale_factor = 4.0

    return config


def validate_ai_config(config: AIEnhancementConfig) -> bool:
    """
    Validate AI enhancement configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate thresholds
        assert 0 <= config.ultra_low_quality_threshold <= 1, "Invalid ultra low quality threshold"
        assert 0 <= config.low_quality_threshold <= 1, "Invalid low quality threshold"
        assert 0 <= config.noise_threshold <= 1, "Invalid noise threshold"

        # Validate enhancement parameters
        assert config.sr_scale_factor > 1, "Super-resolution scale factor must be > 1"
        assert 0 <= config.denoising_strength <= 1, "Invalid denoising strength"
        assert config.contrast_boost_factor > 0, "Contrast boost factor must be positive"
        assert config.gamma_correction > 0, "Gamma correction must be positive"

        # Validate binarization parameters
        assert config.sauvola_k > 0, "Sauvola k parameter must be positive"
        assert config.sauvola_window_size > 0, "Sauvola window size must be positive"
        assert config.sauvola_window_size % 2 == 1, "Sauvola window size must be odd"

        # Validate block sizes
        for block_size in config.adaptive_block_sizes:
            assert block_size > 0, "Block size must be positive"
            assert block_size % 2 == 1, "Block size must be odd"

        # Validate component filtering
        assert config.min_component_area > 0, "Minimum component area must be positive"
        assert 0 < config.max_component_area_ratio <= 1, "Invalid max component area ratio"

        # Validate performance settings
        assert config.processing_timeout > 0, "Processing timeout must be positive"
        assert config.memory_limit_mb > 0, "Memory limit must be positive"

        return True

    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False


def print_ai_config_summary(config: AIEnhancementConfig):
    """Print a summary of the AI enhancement configuration."""
    print("AI Enhancement Configuration Summary")
    print("=" * 50)
    print(f"GPU Acceleration: {'Enabled' if config.use_gpu else 'Disabled'}")
    print(f"Target DPI: {config.target_dpi}")
    print(f"Super-resolution: {'Enabled' if config.enable_super_resolution else 'Disabled'}")
    if config.enable_super_resolution:
        print(f"  Scale factor: {config.sr_scale_factor}x")
    print(f"AI Denoising: {'Enabled' if config.enable_ai_denoising else 'Disabled'}")
    if config.enable_ai_denoising:
        print(f"  Strength: {config.denoising_strength}")
    print(f"Text Enhancement: {'Enabled' if config.enable_text_detection else 'Disabled'}")
    if config.enable_text_detection:
        print(f"  Strength: {config.text_enhancement_strength}")
    print(f"Multi-method Binarization: {'Enabled' if config.multi_method_binarization else 'Disabled'}")
    print(f"Debug Mode: {'Enabled' if config.debug_mode else 'Disabled'}")
    print("=" * 50)

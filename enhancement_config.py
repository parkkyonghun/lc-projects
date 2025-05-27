"""
Configuration settings for image enhancement pipeline.

This module contains configurable parameters for different image enhancement
techniques, allowing fine-tuning for specific use cases and image types.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ResizeConfig:
    """Configuration for image resizing."""
    target_dpi: int = 300
    min_width: int = 800
    min_height: int = 600
    interpolation_method: str = "LANCZOS"  # LANCZOS, BICUBIC, BILINEAR


@dataclass
class SkewCorrectionConfig:
    """Configuration for skew detection and correction."""
    enable_hough_method: bool = True
    enable_projection_method: bool = True
    min_angle_threshold: float = 0.1  # Minimum angle to apply correction
    max_angle_range: float = 30.0  # Maximum angle to consider
    hough_threshold: int = 100
    projection_angle_step: float = 0.5
    projection_angle_range: Tuple[float, float] = (-5.0, 5.0)


@dataclass
class LightingConfig:
    """Configuration for lighting enhancement."""
    shadow_removal_kernel_size: Tuple[int, int] = (20, 20)
    background_blur_size: int = 19
    normalization_epsilon: float = 1e-6


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction."""
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    nlm_h: float = 10.0
    nlm_template_window_size: int = 7
    nlm_search_window_size: int = 21


@dataclass
class ContrastConfig:
    """Configuration for contrast enhancement."""
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    gamma_correction: float = 1.2


@dataclass
class BinarizationConfig:
    """Configuration for image binarization."""
    adaptive_block_size: int = 11
    adaptive_c_constant: int = 2
    combine_methods: bool = True  # Combine multiple binarization methods


@dataclass
class MorphologyConfig:
    """Configuration for morphological operations."""
    noise_removal_kernel_size: Tuple[int, int] = (2, 2)
    noise_removal_iterations: int = 1
    text_connection_kernel_size: Tuple[int, int] = (3, 1)
    text_connection_iterations: int = 1
    text_strengthen_kernel_size: Tuple[int, int] = (1, 2)
    text_strengthen_iterations: int = 1


@dataclass
class QualityAssessmentConfig:
    """Configuration for quality assessment."""
    sharpness_method: str = "laplacian"  # laplacian, sobel, variance
    noise_estimation_method: str = "high_frequency"  # high_frequency, wavelet
    edge_detection_low_threshold: int = 50
    edge_detection_high_threshold: int = 150


@dataclass
class EnhancementConfig:
    """Main configuration class combining all enhancement settings."""
    resize: ResizeConfig = ResizeConfig()
    skew_correction: SkewCorrectionConfig = SkewCorrectionConfig()
    lighting: LightingConfig = LightingConfig()
    noise_reduction: NoiseReductionConfig = NoiseReductionConfig()
    contrast: ContrastConfig = ContrastConfig()
    binarization: BinarizationConfig = BinarizationConfig()
    morphology: MorphologyConfig = MorphologyConfig()
    quality_assessment: QualityAssessmentConfig = QualityAssessmentConfig()

    # Global settings
    debug_mode: bool = False
    save_intermediate_steps: bool = False
    output_format: str = "PNG"  # PNG, JPEG, TIFF


# Predefined configurations for different use cases
KHMER_ID_CARD_CONFIG = EnhancementConfig(
    resize=ResizeConfig(
        target_dpi=300,
        min_width=1000,
        min_height=700
    ),
    skew_correction=SkewCorrectionConfig(
        min_angle_threshold=0.2,
        max_angle_range=15.0
    ),
    contrast=ContrastConfig(
        clahe_clip_limit=2.5,
        gamma_correction=1.1
    ),
    binarization=BinarizationConfig(
        adaptive_block_size=13,
        adaptive_c_constant=3
    )
)

GENERAL_DOCUMENT_CONFIG = EnhancementConfig(
    resize=ResizeConfig(
        target_dpi=300,
        min_width=800,
        min_height=600
    ),
    skew_correction=SkewCorrectionConfig(
        min_angle_threshold=0.1,
        max_angle_range=30.0
    ),
    contrast=ContrastConfig(
        clahe_clip_limit=3.0,
        gamma_correction=1.2
    )
)

LOW_QUALITY_IMAGE_CONFIG = EnhancementConfig(
    noise_reduction=NoiseReductionConfig(
        bilateral_d=11,
        nlm_h=15.0
    ),
    contrast=ContrastConfig(
        clahe_clip_limit=4.0,
        gamma_correction=1.3
    ),
    morphology=MorphologyConfig(
        noise_removal_iterations=2,
        text_connection_iterations=2
    )
)

HIGH_RESOLUTION_CONFIG = EnhancementConfig(
    resize=ResizeConfig(
        target_dpi=400,
        min_width=1200,
        min_height=900
    ),
    noise_reduction=NoiseReductionConfig(
        bilateral_d=7,
        nlm_h=8.0
    ),
    binarization=BinarizationConfig(
        adaptive_block_size=15
    )
)


def get_config_by_name(config_name: str) -> EnhancementConfig:
    """
    Get predefined configuration by name.

    Args:
        config_name: Name of the configuration

    Returns:
        EnhancementConfig instance

    Raises:
        ValueError: If config_name is not found
    """
    configs = {
        "khmer_id": KHMER_ID_CARD_CONFIG,
        "general": GENERAL_DOCUMENT_CONFIG,
        "low_quality": LOW_QUALITY_IMAGE_CONFIG,
        "high_resolution": HIGH_RESOLUTION_CONFIG,
        "default": EnhancementConfig()
    }

    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return configs[config_name]


def create_custom_config(**kwargs) -> EnhancementConfig:
    """
    Create a custom configuration with overrides.

    Args:
        **kwargs: Configuration overrides

    Returns:
        EnhancementConfig with custom settings
    """
    config = EnhancementConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested configuration updates
            parts = key.split('.')
            if len(parts) == 2:
                section, param = parts
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)

    return config


def validate_config(config: EnhancementConfig) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate resize config
        assert config.resize.target_dpi > 0, "Target DPI must be positive"
        assert config.resize.min_width > 0, "Minimum width must be positive"
        assert config.resize.min_height > 0, "Minimum height must be positive"

        # Validate skew correction
        assert 0 <= config.skew_correction.min_angle_threshold <= 90, "Invalid angle threshold"
        assert 0 < config.skew_correction.max_angle_range <= 90, "Invalid angle range"

        # Validate contrast settings
        assert config.contrast.clahe_clip_limit > 0, "CLAHE clip limit must be positive"
        assert config.contrast.gamma_correction > 0, "Gamma correction must be positive"

        # Validate binarization
        assert config.binarization.adaptive_block_size % 2 == 1, "Block size must be odd"
        assert config.binarization.adaptive_block_size >= 3, "Block size must be >= 3"

        return True

    except AssertionError as e:
        print(f"Configuration validation failed: {e}")
        return False


def print_config_summary(config: EnhancementConfig):
    """Print a summary of the configuration settings."""
    print("Image Enhancement Configuration Summary")
    print("=" * 50)
    print(f"Target DPI: {config.resize.target_dpi}")
    print(f"Minimum size: {config.resize.min_width}x{config.resize.min_height}")
    print(f"Skew correction threshold: {config.skew_correction.min_angle_threshold}°")
    print(f"CLAHE clip limit: {config.contrast.clahe_clip_limit}")
    print(f"Gamma correction: {config.contrast.gamma_correction}")
    print(f"Adaptive threshold block size: {config.binarization.adaptive_block_size}")
    print(f"Debug mode: {config.debug_mode}")
    print("=" * 50)

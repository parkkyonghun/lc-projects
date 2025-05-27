# Image Enhancement for Tesseract OCR

This document describes the comprehensive image enhancement pipeline designed to optimize images for Tesseract OCR, with special focus on Cambodian ID cards and Khmer script recognition.

## Overview

The enhanced preprocessing pipeline significantly improves OCR accuracy through a multi-stage approach that addresses common image quality issues:

- **Skew correction** - Automatically detects and corrects document rotation
- **Perspective correction** - Fixes perspective distortion in photographed documents
- **Shadow removal** - Eliminates shadows and uneven lighting
- **Advanced noise reduction** - Removes noise while preserving text edges
- **Optimal binarization** - Converts to black/white using multiple methods
- **Text-specific morphological operations** - Cleans up text for better recognition

## Key Improvements

### 1. **Enhanced DPI Handling**
- Automatically resizes images to optimal 300 DPI for Tesseract
- Ensures minimum image dimensions for better text recognition
- Uses high-quality LANCZOS resampling

### 2. **Advanced Skew Detection**
- **Hough Line Transform**: Detects text lines and calculates skew angle
- **Projection Profile**: Fallback method for complex layouts
- Automatic rotation with sub-pixel accuracy

### 3. **Perspective Correction**
- Detects document boundaries using contour analysis
- Applies perspective transformation to create rectangular view
- Particularly useful for photographed ID cards

### 4. **Intelligent Shadow Removal**
- Uses morphological operations to estimate background
- Normalizes lighting across the entire image
- Preserves text contrast while removing shadows

### 5. **Multi-Method Binarization**
- Combines Adaptive Gaussian, Adaptive Mean, and Otsu thresholding
- Intelligent method selection based on local image characteristics
- Optimized for both printed and handwritten text

### 6. **Khmer Script Optimization**
- Special parameters tuned for Khmer character recognition
- Preserves complex character structures and diacritics
- Optimized morphological operations for script characteristics

## Usage

### Basic Usage

```python
from image_enhancement_utils import create_preprocessing_pipeline
from PIL import Image

# Create enhancer
enhancer = create_preprocessing_pipeline(target_dpi=300, debug=False)

# Load and enhance image
image = Image.open("cambodian_id.jpg")
enhanced_image = enhancer.enhance_for_ocr(image)

# Use with Tesseract
import pytesseract
text = pytesseract.image_to_string(enhanced_image, lang='khm+eng')
```

### With Configuration

```python
from enhancement_config import get_config_by_name, KHMER_ID_CARD_CONFIG
from image_enhancement_utils import ImageEnhancer

# Use predefined configuration for Khmer ID cards
config = get_config_by_name("khmer_id")
enhancer = ImageEnhancer(target_dpi=config.resize.target_dpi, debug=True)

enhanced_image = enhancer.enhance_for_ocr(image)
```

### API Usage

The enhanced preprocessing is integrated into the OCR API:

```bash
# Use enhanced preprocessing (default)
curl -X POST "http://localhost:8000/ocr/idcard?enhanced_preprocessing=true" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@cambodian_id.jpg"

# Use legacy preprocessing
curl -X POST "http://localhost:8000/ocr/idcard?enhanced_preprocessing=false" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@cambodian_id.jpg"
```

## Testing and Validation

### Test Single Image

```bash
python test_image_enhancement.py single input_image.jpg --output enhanced_output --debug
```

### Batch Processing

```bash
python test_image_enhancement.py batch input_directory --output batch_output
```

### OCR Comparison

```bash
python test_image_enhancement.py ocr input_image.jpg --output comparison_results
```

## Configuration Options

### Predefined Configurations

- **`khmer_id`**: Optimized for Cambodian ID cards
- **`general`**: General document processing
- **`low_quality`**: For poor quality images with noise
- **`high_resolution`**: For high-resolution scanned documents

### Custom Configuration

```python
from enhancement_config import create_custom_config

config = create_custom_config(
    resize_target_dpi=400,
    contrast_clahe_clip_limit=4.0,
    binarization_adaptive_block_size=15
)
```

## Quality Assessment

The pipeline includes comprehensive quality metrics:

```python
from image_enhancement_utils import assess_image_quality_comprehensive
import numpy as np

img_array = np.array(image.convert('L'))
metrics = assess_image_quality_comprehensive(img_array)

print(f"Sharpness: {metrics['sharpness']:.2f}")
print(f"Contrast: {metrics['std_contrast']:.2f}")
print(f"Text ratio: {metrics['text_ratio']:.2f}")
```

## Performance Considerations

### Processing Time
- Enhanced pipeline: ~2-5 seconds per image (depending on size)
- Legacy pipeline: ~0.5-1 second per image
- Trade-off between speed and accuracy

### Memory Usage
- Processes images in-memory with OpenCV
- Peak memory usage: ~3-4x original image size
- Automatic cleanup of intermediate results

### Optimization Tips
1. **Batch Processing**: Process multiple images to amortize setup costs
2. **Debug Mode**: Disable debug mode in production for better performance
3. **Image Size**: Very large images (>4000px) may benefit from pre-scaling
4. **Configuration**: Use appropriate config for your use case

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install opencv-python pillow numpy
   ```

2. **Poor Results on Handwritten Text**
   - Use `low_quality` configuration
   - Increase noise reduction parameters
   - Consider different PSM modes in Tesseract

3. **Slow Processing**
   - Disable debug mode
   - Reduce target DPI for faster processing
   - Use batch processing for multiple images

4. **Over-processing**
   - Reduce CLAHE clip limit
   - Decrease noise reduction iterations
   - Use legacy preprocessing for high-quality scans

### Debug Mode

Enable debug mode to save intermediate processing steps:

```python
enhancer = create_preprocessing_pipeline(debug=True)
enhanced_image = enhancer.enhance_for_ocr(image)
# Check debug_01_grayscale.png, debug_02_deskewed.png, etc.
```

## Best Practices

1. **Image Quality**: Start with highest quality source images possible
2. **Lighting**: Ensure even lighting when capturing images
3. **Resolution**: Use at least 300 DPI for optimal results
4. **Testing**: Test with representative sample images
5. **Configuration**: Fine-tune parameters for your specific use case

## Integration with Existing Code

The enhanced preprocessing is backward-compatible:

```python
# Existing code continues to work
processed_image = preprocess_image(image)

# New enhanced preprocessing
enhancer = create_preprocessing_pipeline()
enhanced_image = enhancer.enhance_for_ocr(image)
```

## Future Enhancements

Planned improvements include:
- Machine learning-based quality assessment
- Adaptive parameter selection based on image characteristics
- Support for additional scripts and languages
- Real-time processing optimizations
- GPU acceleration for batch processing

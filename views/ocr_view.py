from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from controllers.ocr_controller import process_cambodian_id_ocr
from schemas.ocr import CambodianIDCardOCRResult

router = APIRouter(prefix="/ocr", tags=["ocr"])

@router.post("/idcard", response_model=CambodianIDCardOCRResult)
async def ocr_idcard(
    file: UploadFile = File(...),
    enhanced_preprocessing: bool = Query(True, description="Use enhanced preprocessing pipeline"),
    ai_enhancement: bool = Query(False, description="Use AI-powered enhancement for ultra-low quality images"),
    extreme_enhancement: bool = Query(False, description="Use extreme enhancement for severely damaged images"),
    enhancement_mode: str = Query("auto", description="AI enhancement mode: auto, ultra_low_quality, low_quality, khmer_optimized, high_performance"),
    robust_parsing: bool = Query(True, description="Use robust parsing for poor quality OCR text")
):
    """
    Extract text from Cambodian ID card images using advanced OCR with AI enhancement.

    Args:
        file: Image file (JPEG, PNG, etc.)
        enhanced_preprocessing: Whether to use the enhanced preprocessing pipeline
        ai_enhancement: Whether to use AI-powered enhancement for ultra-low quality images
        extreme_enhancement: Whether to use extreme enhancement for severely damaged images
        enhancement_mode: AI enhancement mode for different quality levels and use cases
        robust_parsing: Whether to use robust parsing for poor quality OCR text

    Returns:
        Structured OCR results with extracted fields

    Enhancement Modes:
        - auto: Automatically select best enhancement based on image quality
        - ultra_low_quality: Maximum enhancement for extremely poor quality images
        - low_quality: Balanced enhancement for low quality images
        - khmer_optimized: Optimized specifically for Khmer script
        - high_performance: Fast processing with good quality

    Enhancement Levels (in order of aggressiveness):
        1. enhanced_preprocessing: Standard enhanced preprocessing
        2. ai_enhancement: AI-powered enhancement with deep learning
        3. extreme_enhancement: Most aggressive enhancement for severely damaged images
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Validate enhancement mode
    valid_modes = ["auto", "ultra_low_quality", "low_quality", "khmer_optimized", "high_performance"]
    if enhancement_mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid enhancement_mode. Must be one of: {', '.join(valid_modes)}"
        )

    result = await process_cambodian_id_ocr(
        file,
        use_enhanced_preprocessing=enhanced_preprocessing,
        use_ai_enhancement=ai_enhancement,
        use_extreme_enhancement=extreme_enhancement,
        enhancement_mode=enhancement_mode,
        use_robust_parsing=robust_parsing
    )
    return result

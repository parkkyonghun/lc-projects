#!/usr/bin/env python3
"""
Simple test script to evaluate the Cambodian ID OCR functionality with a single image.
"""

import os
import asyncio
import json
from fastapi import UploadFile
from controllers.ocr_controller import process_cambodian_id_ocr
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_Simple_Test")

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

async def test_single_image(image_path: str, ground_truth_file: Optional[str] = None):
    """Test OCR on a single image and compare with ground truth if available."""
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return
    
    logger.info(f"Processing image: {image_path}")
    
    # Create mock upload file
    upload_file = SimpleUploadFile(image_path)
    
    try:
        # Process with OCR controller
        result = await process_cambodian_id_ocr(upload_file)
        result_dict = result.dict()
        
        # Display OCR results
        logger.info("OCR Results:")
        for key, value in result_dict.items():
            if key not in ["raw_khmer", "raw_english"]:  # Skip raw text for cleaner output
                logger.info(f"  {key}: {value}")
        
        # Compare with ground truth if available
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            # Get image filename to match with ground truth
            img_filename = os.path.basename(image_path)
            
            if img_filename in ground_truth:
                gt_data = ground_truth[img_filename]
                logger.info("\nComparison with ground truth:")
                
                # Create evaluation data
                eval_data = []
                for field in ["id_number", "full_name", "name_kh", "name_en", "date_of_birth", "gender"]:
                    pred_value = str(result_dict.get(field) or "")
                    gt_value = str(gt_data.get(field) or "")
                    
                    match = "✓" if pred_value.lower() == gt_value.lower() else "✗"
                    logger.info(f"  {field}: {pred_value} vs {gt_value} {match}")
                    
                    # Add to evaluation data
                    if gt_value:
                        eval_data.append(f"{pred_value}\t{gt_value}")
                
                # Write evaluation data to file
                with open('ocr_evaluation.tsv', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(eval_data))
                
                logger.info("\nEvaluation file created: ocr_evaluation.tsv")
                logger.info("Now you can run: python evaluate.py --input ocr_evaluation.tsv")
            else:
                logger.warning(f"No ground truth found for {img_filename}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")

async def main():
    image_path = "id_card.jpg"  # Hardcoded for simplicity
    ground_truth_file = "sample_ground_truth.json"
    
    await test_single_image(image_path, ground_truth_file)

if __name__ == "__main__":
    asyncio.run(main())

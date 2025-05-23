#!/usr/bin/env python3
"""
Test script to evaluate the enhanced Cambodian ID OCR functionality.
Processes images and creates evaluation data for evaluate.py.
"""

import os
import argparse
from fastapi import UploadFile
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from controllers.ocr_controller import process_cambodian_id_ocr
import json
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OCR_Evaluation")

class MockUploadFile(UploadFile):
    """Mock UploadFile to simulate FastAPI file upload for testing."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.content_type = "image/jpeg"  # Assume jpeg for simplicity
        self._file = None

    async def read(self) -> bytes:
        """Read file as bytes."""
        with open(self.file_path, "rb") as f:
            return f.read()

async def process_image(image_path: str) -> Dict[str, Any]:
    """Process a single image and return OCR results."""
    logger.info(f"Processing image: {image_path}")
    
    # Create mock UploadFile object
    upload_file = MockUploadFile(image_path)
    
    # Process with our OCR controller
    try:
        result = await process_cambodian_id_ocr(upload_file)
        return result.dict()
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return {"error": str(e)}

async def process_images(image_path: str) -> List[Dict[str, Any]]:
    """Process images from a directory or a single image file."""
    image_files = []
    
    # Check if the path is a file or directory
    if os.path.isfile(image_path):
        # Single image file
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files = [image_path]
            logger.info(f"Processing single image file: {image_path}")
        else:
            logger.warning(f"File {image_path} is not a supported image format")
    elif os.path.isdir(image_path):
        # Directory of images
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
        logger.info(f"Found {len(image_files)} images in directory {image_path}")
    else:
        logger.error(f"Path does not exist: {image_path}")
    
    # Process each image
    results = []
    for img_path in image_files:
        result = await process_image(img_path)
        result['image_path'] = img_path  # Add image path for reference
        results.append(result)
    
    return results

def create_evaluation_file(results: List[Dict[str, Any]], 
                          ground_truth: Dict[str, Dict[str, str]],
                          output_file: str):
    """Create a tab-separated evaluation file for evaluate.py."""
    fields = ["id_number", "full_name", "name_kh", "name_en", "date_of_birth", "gender"]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # For each processed image
        for result in results:
            image_id = os.path.basename(result['image_path'])
            
            # Skip if ground truth not available
            if image_id not in ground_truth:
                logger.warning(f"No ground truth for {image_id}, skipping")
                continue
            
            # For each field, write prediction & ground truth
            for field in fields:
                if field in result and field in ground_truth[image_id]:
                    prediction = str(result[field] or "")
                    label = str(ground_truth[image_id][field] or "")
                    f.write(f"{prediction}\t{label}\n")

def load_ground_truth(ground_truth_file: str) -> Dict[str, Dict[str, str]]:
    """Load ground truth data from a JSON file."""
    if not os.path.exists(ground_truth_file):
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        return {}
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main():
    parser = argparse.ArgumentParser(description='Test Cambodian ID OCR and prepare evaluation data')
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Directory containing ID card images')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='JSON file with ground truth labels')
    parser.add_argument('--output', type=str, default='ocr_evaluation.tsv',
                        help='Output file for evaluation (tab-separated)')
    
    args = parser.parse_args()
    
    # Process all images
    results = await process_images(args.image_dir)
    
    # Save raw results for inspection
    with open('ocr_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Create evaluation file
    create_evaluation_file(results, ground_truth, args.output)
    
    logger.info(f"Evaluation file created: {args.output}")
    logger.info(f"Now you can run: python evaluate.py --input {args.output}")

if __name__ == "__main__":
    asyncio.run(main())

import os
import json
from optimized_khmer_ocr import preprocess_image_for_khmer, extract_khmer_text

def run_benchmark_on_task(task_directory, results_output_file, psm_to_use):
    """
    Runs OCR on all images in a benchmark task directory and saves results.
    """
    print(f"Starting benchmark for task: {task_directory}")
    print(f"Using PSM mode: {psm_to_use}")
    print(f"Saving results to: {results_output_file}")

    image_files = [f for f in os.listdir(task_directory) if f.lower().endswith(('.tiff', '.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {task_directory}")
        return

    with open(results_output_file, 'w', encoding='utf-8') as outfile:
        for image_filename in image_files:
            base_filename, _ = os.path.splitext(image_filename)
            image_path = os.path.join(task_directory, image_filename)
            json_path = os.path.join(task_directory, base_filename + '.json')

            if not os.path.exists(json_path):
                print(f"Warning: JSON label file not found for {image_filename}, skipping.")
                continue

            print(f"Processing image: {image_filename}...")

            try:
                # 1. Preprocess the image
                processed_data = preprocess_image_for_khmer(image_path)
                final_image_to_ocr = processed_data['final'] 

                # 2. Extract text using OCR
                ocr_prediction_raw = extract_khmer_text(final_image_to_ocr, psm=psm_to_use)
                
                # Clean OCR prediction: replace multiple spaces/newlines/tabs with a single space
                ocr_prediction_cleaned = ' '.join(ocr_prediction_raw.split()).strip()

                # 3. Load and process ground truth labels from JSON
                with open(json_path, 'r', encoding='utf-8') as f_json:
                    label_data = json.load(f_json)
                
                ground_truth_parts = []
                if 'shapes' in label_data and isinstance(label_data['shapes'], list):
                    for shape in label_data['shapes']:
                        if 'label' in shape and isinstance(shape['label'], str):
                            # Replace internal tabs in label with spaces
                            cleaned_label_part = ' '.join(shape['label'].split()).strip()
                            ground_truth_parts.append(cleaned_label_part)
                
                combined_ground_truth = ' '.join(ground_truth_parts).strip()
                
                # 4. Write to output file
                outfile.write(f"{ocr_prediction_cleaned}\t{combined_ground_truth}\n")

            except Exception as e:
                print(f"Error processing {image_filename}: {e}")
                outfile.write(f"ERROR_PROCESSING_IMAGE_{image_filename}\tERROR_PROCESSING_LABEL\n")
        
    print(f"Benchmark processing finished for task: {task_directory}")
    print(f"Results saved to: {results_output_file}")

if __name__ == "__main__":
    # Path to the specific task directory to benchmark
    target_task_dir = r"C:\Users\pkflutter\Downloads\khob-level-1\khob-level-1\task-1"
    
    # PSM modes to evaluate
    psm_modes = {
        3: 'Fully automatic page segmentation, but no OSD',
        4: 'Assume a single column of text of variable sizes',
        6: 'Assume a single uniform block of text',
        11: 'Sparse text. Find as much text as possible in no particular order'
    }
    
    # Create output directory for results
    benchmark_output_dir = "benchmark_results" 
    if not os.path.exists(benchmark_output_dir):
        os.makedirs(benchmark_output_dir)
    
    task_name = os.path.basename(target_task_dir)
    
    # Check if task directory exists
    if not os.path.isdir(target_task_dir):
        print(f"Error: Task directory not found: {target_task_dir}")
        print("Please ensure the 'target_task_dir' variable is set correctly.")
    else:
        # Run benchmark for each PSM mode
        result_files = {}
        
        print(f"Running benchmarks for all PSM modes on {task_name}...\n")
        
        for psm, desc in psm_modes.items():
            print(f"Processing with PSM {psm}: {desc}")
            
            # Set up output filename for this PSM mode
            output_filename = f"results_{task_name}_psm{psm}.txt"
            full_output_path = os.path.join(benchmark_output_dir, output_filename)
            result_files[psm] = full_output_path
            
            # Run the benchmark for this PSM mode
            run_benchmark_on_task(
                task_directory=target_task_dir,
                results_output_file=full_output_path,
                psm_to_use=psm
            )
            print(f"Completed PSM {psm}\n")
        
        # Print summary and next steps
        print("\n============== BENCHMARK COMPLETE ==============\n")
        print("Results files created:")
        for psm, filepath in result_files.items():
            print(f"PSM {psm}: {os.path.abspath(filepath)}")
        
        print("\nTo evaluate these results with the benchmark toolkit:")
        print("1. For each result file, run:")
        print("   python evaluate.py --input [RESULT_FILE_PATH]")
        print("\nExample for PSM 6:")
        print(f"   python evaluate.py --input {os.path.abspath(result_files[6])}")
        print("\nCompare the metrics for each PSM mode to determine which performs best.")

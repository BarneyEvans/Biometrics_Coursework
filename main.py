"""
Main script to run the complete body shape extraction pipeline.
Steps:
1. Detect and crop persons using YOLO
2. Remove green screen background
"""

import os
import argparse
from person_detector import load_yolo_model, process_directory as detect_persons
from green_screen_remover import process_directory as remove_green_screen

def create_directories(base_dir):
    """
    Create the necessary directory structure for the pipeline.
    
    Args:
        base_dir: Base directory for all outputs
        
    Returns:
        dict: Dictionary containing paths to all directories
    """
    dirs = {
        'base': base_dir,
        'person_detection': os.path.join(base_dir, 'person_detection'),
        'train_person': os.path.join(base_dir, 'person_detection', 'train'),
        'test_person': os.path.join(base_dir, 'person_detection', 'test'),
        'final_output': os.path.join(base_dir, 'final_output'),
        'train_final': os.path.join(base_dir, 'final_output', 'train'),
        'test_final': os.path.join(base_dir, 'final_output', 'test')
    }
    
    # Create all directories
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
        
    return dirs

def main():
    parser = argparse.ArgumentParser(description='Body Shape Extraction Pipeline')
    parser.add_argument('--train_dir', default='Model_Images/training', 
                        help='Directory containing training images')
    parser.add_argument('--test_dir', default='Model_Images/test', 
                        help='Directory containing test images')
    parser.add_argument('--output_dir', default='Processed_Images', 
                        help='Base directory for all outputs')
    parser.add_argument('--calibrate_green', action='store_true',
                        help='Run green screen calibration on a sample image')
    parser.add_argument('--sample_image', 
                        help='Sample image for green screen calibration')
    parser.add_argument('--skip_detection', action='store_true',
                        help='Skip person detection step (use if already done)')
    parser.add_argument('--skip_green_removal', action='store_true',
                        help='Skip green screen removal step (use if already done)')
    
    args = parser.parse_args()
    
    # Create directory structure
    dirs = create_directories(args.output_dir)
    
    # Run green screen calibration if requested
    if args.calibrate_green:
        if args.sample_image:
            from green_screen_remover import adjust_green_range
            print("Running green screen calibration...")
            adjust_green_range(args.sample_image)
            return
        else:
            print("Error: --sample_image is required for calibration")
            return
    
    # Step 1: Person Detection
    if not args.skip_detection:
        print("\n=== Step 1: Person Detection ===")
        model = load_yolo_model()
        
        print("\nProcessing training images...")
        detect_persons(model, args.train_dir, dirs['train_person'])
        
        print("\nProcessing test images...")
        detect_persons(model, args.test_dir, dirs['test_person'])
    else:
        print("\n=== Skipping Person Detection ===")
    
    # Step 2: Green Screen Removal
    if not args.skip_green_removal:
        print("\n=== Step 2: Green Screen Removal ===")
        
        print("\nProcessing training images...")
        remove_green_screen(dirs['train_person'], dirs['train_final'])
        
        print("\nProcessing test images...")
        remove_green_screen(dirs['test_person'], dirs['test_final'])
    else:
        print("\n=== Skipping Green Screen Removal ===")
    
    print("\n=== Processing Complete! ===")
    print(f"Final outputs are available in:\n- Training: {dirs['train_final']}\n- Testing: {dirs['test_final']}")

if __name__ == "__main__":
    main()
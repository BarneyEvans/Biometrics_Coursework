import os
import argparse
from person_detector import load_yolo_model, process_directory as detect_persons

def clear_directory(directory):
    """
    Clears all files in a directory if it exists, otherwise creates it.
    
    Args:
        directory (str): Path to the directory to clear.
    """
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Person Detection Pipeline')
    parser.add_argument('--input_dir', required=True, help='Path to input images (train/test directory)')
    parser.add_argument('--output_dir', required=True, help='Path to store YOLO-detected images')
    
    args = parser.parse_args()
    
    # Load YOLO model
    model = load_yolo_model()
    
    # Clear output directory before processing
    clear_directory(args.output_dir)

    # Run person detection
    print("\n=== Running Person Detection ===")
    detect_persons(model, args.input_dir, args.output_dir)
    
    print("\n=== Processing Complete! ===")
    print(f"Processed images saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

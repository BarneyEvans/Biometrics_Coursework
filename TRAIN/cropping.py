"""
Person detection module using YOLOv8.
This script detects people in images from the training folder and saves the cropped regions.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_yolo_model():
    """
    Load YOLOv8 model and configure it to detect only people.
    
    Returns:
        model: YOLOv8 model object
    """
    # Load YOLOv8 large model
    model = YOLO("yolov8l.pt")
    return model

def detect_person(model, image_path):
    """
    Detect persons in an image and return the image with bounding box.
    
    Args:
        model: YOLOv8 model
        image_path: Path to the input image
        
    Returns:
        tuple: (image, bounding_box) where bounding_box is (x1, y1, x2, y2)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None, None
    
    # Convert BGR to RGB for YOLO
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(rgb_img)[0]  # YOLOv8 returns a list; get first result

    # Get detection results (x1, y1, x2, y2, confidence, class)
    detections = results.boxes.data.cpu().numpy()

    # Filter for person detections (class 0 in COCO dataset)
    person_detections = [d for d in detections if int(d[5]) == 0]

    if not person_detections:
        print(f"No person detected in {image_path}")
        return img, None
    
    # Get the detection with highest confidence
    best_detection = sorted(person_detections, key=lambda x: x[4], reverse=True)[0]
    x1, y1, x2, y2 = map(int, best_detection[:4])
    
    return img, (x1, y1, x2, y2)

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
        os.makedirs(directory, exist_ok=True)

def process_training_images(model, input_dir="IMAGES\\Original_Images\\training", 
                            output_dir="IMAGES\\Processed_Images\\Cropped\\Train"):
    """
    Process all training images to detect and crop persons.
    
    Args:
        model: YOLOv8 model
        input_dir: Directory containing input training images
        output_dir: Directory to save cropped images
    """
    # Clear output directory before processing
    clear_directory(output_dir)
    
    # Get all image files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n=== Processing {len(files)} images from training set ===")
    
    for idx, file in enumerate(files):
        print(f"Detecting person in {idx+1}/{len(files)}: {file}")
        
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Detect person
        image, bbox = detect_person(model, input_path)
        
        if image is not None and bbox is not None:
            # Crop the image to the bounding box with a small margin
            x1, y1, x2, y2 = bbox
            
            # Add a small margin (5% of height/width)
            h, w = image.shape[:2]
            margin_x = int(0.05 * (x2 - x1))
            margin_y = int(0.05 * (y2 - y1))
            
            # Apply margins with bounds checking
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            # Crop and save
            cropped = image[y1:y2, x1:x2]
            cv2.imwrite(output_path, cropped)
            print(f"Saved cropped person to {output_path}")
        elif image is not None:
            # Save the original image if no person was detected
            cv2.imwrite(output_path, image)
            print(f"No person detected, saving original image to {output_path}")
        else:
            print(f"Failed to process {input_path}")

def main():
    """Main function to run the person detection pipeline"""
    # Load YOLO model
    print("Loading YOLOv8 model...")
    model = load_yolo_model()
    
    # Process training images with default paths
    process_training_images(model)
    
    print("\n=== Processing Complete! ===")
    print(f"Processed images saved in: IMAGES\\Processed_Images\\Cropped\\Train")

if __name__ == "__main__":
    main()
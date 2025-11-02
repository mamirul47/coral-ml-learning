#!/usr/bin/env python3
"""
Simple Image Cropping Examples
This script demonstrates various ways to crop images based on bounding boxes
"""

import cv2
import numpy as np
from PIL import Image
import os
from image_preprocessor import ImagePreprocessor

def example_basic_cropping():
    """Example of basic image cropping with different bbox formats"""
    print("=== Basic Image Cropping Examples ===")
    
    if not os.path.exists("coral.jpeg"):
        print("coral.jpeg not found, skipping examples")
        return
    
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = preprocessor.load_image("coral.jpeg")
    h, w = image.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Example 1: Crop center square (xyxy format)
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 3
    bbox_xyxy = (center_x - size//2, center_y - size//2, 
                 center_x + size//2, center_y + size//2)
    
    crop1 = preprocessor.crop_image_bbox(image, bbox_xyxy, "xyxy")
    cv2.imwrite("crop_center_xyxy.jpg", crop1)
    print(f"1. Center crop (xyxy): {bbox_xyxy} -> crop_center_xyxy.jpg")
    
    # Example 2: Crop using xywh format
    bbox_xywh = (100, 100, 400, 300)  # x, y, width, height
    crop2 = preprocessor.crop_image_bbox(image, bbox_xywh, "xywh")
    cv2.imwrite("crop_xywh.jpg", crop2)
    print(f"2. XYWH crop: {bbox_xywh} -> crop_xywh.jpg")
    
    # Example 3: Crop using center coordinates
    bbox_cxcywh = (w//2, h//2, 300, 300)  # center_x, center_y, width, height
    crop3 = preprocessor.crop_image_bbox(image, bbox_cxcywh, "cxcywh")
    cv2.imwrite("crop_cxcywh.jpg", crop3)
    print(f"3. Center coords crop: {bbox_cxcywh} -> crop_cxcywh.jpg")
    
    # Example 4: Normalized coordinates (0-1 range)
    bbox_norm = (0.2, 0.2, 0.8, 0.8)  # left, top, right, bottom as ratios
    crop4 = preprocessor.crop_image_bbox(image, bbox_norm, "normalized")
    cv2.imwrite("crop_normalized.jpg", crop4)
    print(f"4. Normalized crop: {bbox_norm} -> crop_normalized.jpg")

def example_multiple_crops():
    """Example of cropping multiple regions from one image"""
    print("\n=== Multiple Crops Example ===")
    
    if not os.path.exists("coral.jpeg"):
        print("coral.jpeg not found, skipping examples")
        return
    
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = preprocessor.load_image("coral.jpeg")
    h, w = image.shape[:2]
    
    # Define multiple bounding boxes
    bboxes = [
        (50, 50, 300, 300),      # Top-left region
        (w-350, 50, w-50, 300),  # Top-right region
        (50, h-350, 300, h-50),  # Bottom-left region
        (w-350, h-350, w-50, h-50)  # Bottom-right region
    ]
    
    # Process multiple crops
    crops = preprocessor.process_multiple_crops(
        "coral.jpeg", bboxes, format_type="xyxy", 
        output_dir="multiple_crops", save_crops=True
    )
    
    print(f"Created {len(crops)} crops in 'multiple_crops' directory")

def example_crop_with_preprocessing():
    """Example of cropping with additional preprocessing"""
    print("\n=== Crop with Preprocessing Example ===")
    
    if not os.path.exists("coral.jpeg"):
        print("coral.jpeg not found, skipping examples")
        return
    
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = preprocessor.load_image("coral.jpeg")
    h, w = image.shape[:2]
    
    # Define crop area
    bbox = (w//4, h//4, 3*w//4, 3*h//4)
    
    # 1. Basic crop
    crop = preprocessor.crop_image_bbox(image, bbox, "xyxy")
    
    # 2. Resize crop to standard size
    resized_crop = preprocessor.resize_image(crop, (224, 224), keep_aspect_ratio=True)
    
    # 3. Enhance the cropped image
    enhanced_crop = preprocessor.adjust_brightness_contrast(resized_crop, brightness=10, contrast=1.2)
    enhanced_crop = preprocessor.apply_histogram_equalization(enhanced_crop)
    
    # Save results
    cv2.imwrite("crop_basic.jpg", crop)
    cv2.imwrite("crop_resized.jpg", resized_crop)
    cv2.imwrite("crop_enhanced.jpg", enhanced_crop)
    
    print("Saved: crop_basic.jpg, crop_resized.jpg, crop_enhanced.jpg")

def example_draw_bboxes():
    """Example of drawing bounding boxes on images"""
    print("\n=== Draw Bounding Boxes Example ===")
    
    if not os.path.exists("coral.jpeg"):
        print("coral.jpeg not found, skipping examples")
        return
    
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = preprocessor.load_image("coral.jpeg")
    h, w = image.shape[:2]
    
    # Define some example bounding boxes
    bboxes = [
        {"bbox": (100, 100, 300, 250), "label": "Region 1", "color": (0, 255, 0)},
        {"bbox": (w-300, 100, w-100, 250), "label": "Region 2", "color": (255, 0, 0)},
        {"bbox": (w//2-100, h//2-50, w//2+100, h//2+50), "label": "Center", "color": (0, 0, 255)}
    ]
    
    # Draw all bounding boxes
    result_image = image.copy()
    for i, bbox_info in enumerate(bboxes):
        result_image = preprocessor.draw_bounding_box(
            result_image, 
            bbox_info["bbox"], 
            format_type="xyxy",
            color=bbox_info["color"],
            thickness=3,
            label=bbox_info["label"]
        )
    
    cv2.imwrite("image_with_bboxes.jpg", result_image)
    print("Saved: image_with_bboxes.jpg")

def example_batch_processing():
    """Example of processing multiple images"""
    print("\n=== Batch Processing Example ===")
    
    # Find all image files in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("No image files found in current directory")
        return
    
    preprocessor = ImagePreprocessor()
    
    print(f"Found {len(image_files)} image(s): {image_files}")
    
    # Process each image
    for image_file in image_files:
        try:
            print(f"\nProcessing: {image_file}")
            
            # Load image
            image = preprocessor.load_image(image_file)
            h, w = image.shape[:2]
            print(f"  Size: {w}x{h}")
            
            # Create a center crop
            crop_size = min(w, h) // 2
            center_x, center_y = w // 2, h // 2
            bbox = (center_x - crop_size//2, center_y - crop_size//2,
                   center_x + crop_size//2, center_y + crop_size//2)
            
            # Crop and resize
            crop = preprocessor.crop_image_bbox(image, bbox, "xyxy")
            resized = preprocessor.resize_image(crop, (224, 224))
            
            # Save processed image
            base_name = os.path.splitext(image_file)[0]
            output_name = f"{base_name}_processed.jpg"
            cv2.imwrite(output_name, resized)
            print(f"  Saved: {output_name}")
            
        except Exception as e:
            print(f"  Error processing {image_file}: {e}")

if __name__ == "__main__":
    print("Image Preprocessing and Cropping Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_cropping()
    example_multiple_crops()
    example_crop_with_preprocessing()
    example_draw_bboxes()
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nGenerated files:")
    print("- crop_*.jpg (various cropping examples)")
    print("- multiple_crops/ (directory with multiple crops)")
    print("- image_with_bboxes.jpg (image with drawn bounding boxes)")
    print("- *_processed.jpg (batch processed images)")
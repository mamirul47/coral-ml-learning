import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from image_preprocessor import ImagePreprocessor

class YOLOCropper:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO-based cropper
        
        Args:
            model_path: Path to YOLO model weights, if None uses YOLOv8n
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use a default YOLOv8 model
            self.model = YOLO('yolov8n.pt')
        
        self.preprocessor = ImagePreprocessor()
    
    def detect_and_crop(self, image_path: str, 
                       confidence_threshold: float = 0.5,
                       output_dir: str = "yolo_crops",
                       save_crops: bool = True,
                       save_annotated: bool = True) -> dict:
        """
        Detect objects using YOLO and crop them
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            output_dir: Directory to save results
            save_crops: Whether to save individual crops
            save_annotated: Whether to save annotated image with boxes
        
        Returns:
            Dictionary with detection results and crop information
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run YOLO detection
        results = self.model(image_path, conf=confidence_threshold)
        
        # Process results
        detections = []
        crops = []
        
        if save_crops or save_annotated:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection_info = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection_info)
                    
                    if save_crops:
                        # Crop the detection
                        try:
                            crop = self.preprocessor.crop_image_bbox(
                                image, (x1, y1, x2, y2), format_type="xyxy"
                            )
                            crops.append(crop)
                            
                            # Save crop
                            crop_filename = f"{base_name}_crop_{i:03d}_{class_name}_{confidence:.2f}.jpg"
                            crop_path = os.path.join(output_dir, crop_filename)
                            cv2.imwrite(crop_path, crop)
                            print(f"Saved crop: {crop_path}")
                            
                        except Exception as e:
                            print(f"Error cropping detection {i}: {e}")
        
        # Save annotated image
        if save_annotated and detections:
            annotated_image = image.copy()
            for detection in detections:
                bbox = detection['bbox']
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                
                annotated_image = self.preprocessor.draw_bounding_box(
                    annotated_image, bbox, format_type="xyxy",
                    color=(0, 255, 0), thickness=2, label=label
                )
            
            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            print(f"Saved annotated image: {annotated_path}")
        
        return {
            'detections': detections,
            'crops': crops,
            'num_detections': len(detections)
        }
    
    def crop_specific_classes(self, image_path: str, 
                             target_classes: list,
                             confidence_threshold: float = 0.5,
                             output_dir: str = "class_crops") -> dict:
        """
        Detect and crop only specific classes
        
        Args:
            image_path: Path to input image
            target_classes: List of class names to crop
            confidence_threshold: Minimum confidence for detections
            output_dir: Directory to save results
        
        Returns:
            Dictionary with filtered detection results
        """
        # Get all detections
        results = self.detect_and_crop(
            image_path, confidence_threshold, 
            output_dir, save_crops=False, save_annotated=False
        )
        
        # Filter for target classes
        filtered_detections = [
            det for det in results['detections'] 
            if det['class_name'] in target_classes
        ]
        
        # Crop filtered detections
        image = cv2.imread(image_path)
        filtered_crops = []
        
        if filtered_detections:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, detection in enumerate(filtered_detections):
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                try:
                    crop = self.preprocessor.crop_image_bbox(
                        image, bbox, format_type="xyxy"
                    )
                    filtered_crops.append(crop)
                    
                    # Save crop
                    crop_filename = f"{base_name}_{class_name}_{i:03d}_{confidence:.2f}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    print(f"Saved {class_name} crop: {crop_path}")
                    
                except Exception as e:
                    print(f"Error cropping {class_name} detection: {e}")
        
        return {
            'detections': filtered_detections,
            'crops': filtered_crops,
            'num_detections': len(filtered_detections)
        }

def main():
    """Example usage of YOLO-based cropping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO-based Object Detection and Cropping")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--classes", type=str, nargs="+", help="Specific classes to crop")
    parser.add_argument("--output", type=str, default="yolo_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        cropper = YOLOCropper(args.model)
        
        if args.classes:
            print(f"Detecting and cropping specific classes: {args.classes}")
            results = cropper.crop_specific_classes(
                args.image, args.classes, args.conf, args.output
            )
        else:
            print("Detecting and cropping all objects")
            results = cropper.detect_and_crop(
                args.image, args.conf, args.output
            )
        
        print(f"\nResults:")
        print(f"Number of detections: {results['num_detections']}")
        for i, detection in enumerate(results['detections']):
            print(f"Detection {i+1}: {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.2f})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("YOLO Cropper - Example Usage:")
        print("\n1. Detect and crop all objects:")
        print("python yolo_cropper.py --image coral.jpeg --conf 0.5")
        print("\n2. Crop specific classes only:")
        print("python yolo_cropper.py --image coral.jpeg --classes person car --conf 0.3")
        print("\n3. Use custom YOLO model:")
        print("python yolo_cropper.py --image coral.jpeg --model path/to/model.pt --conf 0.5")
        
        # Demo if coral image exists
        if os.path.exists("coral.jpeg"):
            print("\nRunning demo with coral.jpeg...")
            try:
                cropper = YOLOCropper()
                results = cropper.detect_and_crop("coral.jpeg", confidence_threshold=0.3)
                print(f"Demo completed! Found {results['num_detections']} detections")
            except Exception as e:
                print(f"Demo error (this is normal if YOLO model isn't available): {e}")
    else:
        main()
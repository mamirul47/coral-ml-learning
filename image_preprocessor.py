import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from typing import List, Tuple, Union, Optional
import argparse

class ImagePreprocessor:
    def __init__(self):
        """Initialize the Image Preprocessor"""
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image using OpenCV"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def load_image_pil(self, image_path: str) -> Image.Image:
        """Load image using PIL"""
        try:
            image = Image.open(image_path)
            return image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not load image from {image_path}: {e}")
    
    def crop_image_bbox(self, image: Union[np.ndarray, Image.Image], 
                       bbox: Tuple[int, int, int, int], 
                       format_type: str = "xyxy") -> Union[np.ndarray, Image.Image]:
        """
        Crop image based on bounding box
        
        Args:
            image: Input image (OpenCV or PIL format)
            bbox: Bounding box coordinates
            format_type: Format of bbox coordinates
                - "xyxy": (x1, y1, x2, y2) - top-left and bottom-right corners
                - "xywh": (x, y, width, height) - top-left corner and dimensions
                - "cxcywh": (center_x, center_y, width, height) - center and dimensions
        
        Returns:
            Cropped image in the same format as input
        """
        if isinstance(image, np.ndarray):
            return self._crop_opencv(image, bbox, format_type)
        elif isinstance(image, Image.Image):
            return self._crop_pil(image, bbox, format_type)
        else:
            raise ValueError("Image must be numpy array (OpenCV) or PIL Image")
    
    def _crop_opencv(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                    format_type: str) -> np.ndarray:
        """Crop OpenCV image"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox, format_type, w, h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box coordinates")
        
        return image[y1:y2, x1:x2]
    
    def _crop_pil(self, image: Image.Image, bbox: Tuple[int, int, int, int], 
                 format_type: str) -> Image.Image:
        """Crop PIL image"""
        w, h = image.size
        x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox, format_type, w, h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box coordinates")
        
        return image.crop((x1, y1, x2, y2))
    
    def _convert_bbox_to_xyxy(self, bbox: Tuple[int, int, int, int], 
                             format_type: str, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Convert different bbox formats to xyxy format"""
        if format_type == "xyxy":
            return bbox
        elif format_type == "xywh":
            x, y, w, h = bbox
            return (x, y, x + w, y + h)
        elif format_type == "cxcywh":
            cx, cy, w, h = bbox
            x1 = int(cx - w // 2)
            y1 = int(cy - h // 2)
            x2 = int(cx + w // 2)
            y2 = int(cy + h // 2)
            return (x1, y1, x2, y2)
        elif format_type == "normalized":
            # Normalized coordinates (0-1) to absolute
            x1, y1, x2, y2 = bbox
            return (int(x1 * img_w), int(y1 * img_h), 
                   int(x2 * img_w), int(y2 * img_h))
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def resize_image(self, image: Union[np.ndarray, Image.Image], 
                    size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> Union[np.ndarray, Image.Image]:
        """
        Resize image to specified size
        
        Args:
            image: Input image
            size: Target size (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
        """
        if isinstance(image, np.ndarray):
            if keep_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = size
                
                # Calculate scaling factor
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Create padded image
                padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return padded
            else:
                return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        
        elif isinstance(image, Image.Image):
            if keep_aspect_ratio:
                image.thumbnail(size, Image.Resampling.LANCZOS)
                # Create new image with padding
                new_image = Image.new('RGB', size, (0, 0, 0))
                x_offset = (size[0] - image.width) // 2
                y_offset = (size[1] - image.height) // 2
                new_image.paste(image, (x_offset, y_offset))
                return new_image
            else:
                return image.resize(size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, 255] range"""
        return (image * 255).astype(np.uint8)
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                  brightness: float = 0, 
                                  contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for better contrast"""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def draw_bounding_box(self, image: Union[np.ndarray, Image.Image], 
                         bbox: Tuple[int, int, int, int], 
                         format_type: str = "xyxy",
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2,
                         label: Optional[str] = None) -> Union[np.ndarray, Image.Image]:
        """
        Draw bounding box on image
        
        Args:
            image: Input image
            bbox: Bounding box coordinates
            format_type: Format of bbox coordinates
            color: Box color (B, G, R) for OpenCV or (R, G, B) for PIL
            thickness: Line thickness
            label: Optional label text
        """
        if isinstance(image, np.ndarray):
            return self._draw_bbox_opencv(image, bbox, format_type, color, thickness, label)
        elif isinstance(image, Image.Image):
            return self._draw_bbox_pil(image, bbox, format_type, color, thickness, label)
    
    def _draw_bbox_opencv(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                         format_type: str, color: Tuple[int, int, int], 
                         thickness: int, label: Optional[str]) -> np.ndarray:
        """Draw bounding box on OpenCV image"""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox, format_type, w, h)
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            font_scale = 0.6
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return img_copy
    
    def _draw_bbox_pil(self, image: Image.Image, bbox: Tuple[int, int, int, int], 
                      format_type: str, color: Tuple[int, int, int], 
                      thickness: int, label: Optional[str]) -> Image.Image:
        """Draw bounding box on PIL image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        x1, y1, x2, y2 = self._convert_bbox_to_xyxy(bbox, format_type, w, h)
        
        # Draw rectangle
        for i in range(thickness):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, fill=None)
        
        # Draw label if provided
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1], fill=color)
            
            # Draw text
            draw.text((x1, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)
        
        return img_copy
    
    def process_multiple_crops(self, image_path: str, 
                              bboxes: List[Tuple[int, int, int, int]], 
                              format_type: str = "xyxy",
                              output_dir: str = "crops",
                              save_crops: bool = True) -> List[np.ndarray]:
        """
        Process multiple crops from a single image
        
        Args:
            image_path: Path to input image
            bboxes: List of bounding boxes
            format_type: Format of bbox coordinates
            output_dir: Directory to save crops
            save_crops: Whether to save crops to disk
        
        Returns:
            List of cropped images
        """
        image = self.load_image(image_path)
        crops = []
        
        if save_crops:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i, bbox in enumerate(bboxes):
            try:
                crop = self.crop_image_bbox(image, bbox, format_type)
                crops.append(crop)
                
                if save_crops:
                    crop_filename = f"{base_name}_crop_{i:03d}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    print(f"Saved crop: {crop_path}")
                    
            except Exception as e:
                print(f"Error processing crop {i}: {e}")
                continue
        
        return crops

def main():
    """Example usage and command-line interface"""
    parser = argparse.ArgumentParser(description="Image Preprocessing and Cropping Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--bbox", type=int, nargs=4, help="Bounding box coordinates (x1 y1 x2 y2)")
    parser.add_argument("--format", type=str, default="xyxy", 
                       choices=["xyxy", "xywh", "cxcywh", "normalized"],
                       help="Bounding box format")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize to width height")
    parser.add_argument("--draw-box", action="store_true", help="Draw bounding box on image")
    
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor()
    
    try:
        # Load image
        image = preprocessor.load_image(args.image)
        print(f"Loaded image: {args.image}, Shape: {image.shape}")
        
        result_image = image.copy()
        
        # Process bounding box if provided
        if args.bbox:
            bbox = tuple(args.bbox)
            print(f"Processing bbox: {bbox} (format: {args.format})")
            
            if args.draw_box:
                # Draw bounding box
                result_image = preprocessor.draw_bounding_box(
                    result_image, bbox, args.format, 
                    color=(0, 255, 0), thickness=2, label="Detection"
                )
            else:
                # Crop image
                result_image = preprocessor.crop_image_bbox(result_image, bbox, args.format)
        
        # Resize if requested
        if args.resize:
            width, height = args.resize
            result_image = preprocessor.resize_image(result_image, (width, height))
            print(f"Resized to: {width}x{height}")
        
        # Save result
        cv2.imwrite(args.output, result_image)
        print(f"Saved result: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Image Preprocessor - Example Usage:")
        print("\n1. Crop image with bounding box:")
        print("python image_preprocessor.py --image coral.jpeg --bbox 100 100 300 300 --format xyxy --output cropped.jpg")
        print("\n2. Draw bounding box on image:")
        print("python image_preprocessor.py --image coral.jpeg --bbox 100 100 300 300 --draw-box --output with_box.jpg")
        print("\n3. Crop and resize:")
        print("python image_preprocessor.py --image coral.jpeg --bbox 100 100 300 300 --resize 224 224 --output resized_crop.jpg")
        
        # Demo with your coral image if it exists
        if os.path.exists("coral.jpeg"):
            print("\nRunning demo with coral.jpeg...")
            preprocessor = ImagePreprocessor()
            
            # Load image
            image = preprocessor.load_image("coral.jpeg")
            h, w = image.shape[:2]
            print(f"Original image size: {w}x{h}")
            
            # Example crop (center quarter of the image)
            bbox = (w//4, h//4, 3*w//4, 3*h//4)
            
            # Crop image
            cropped = preprocessor.crop_image_bbox(image, bbox, "xyxy")
            cv2.imwrite("demo_cropped.jpg", cropped)
            print("Saved demo_cropped.jpg")
            
            # Draw bounding box
            with_box = preprocessor.draw_bounding_box(
                image, bbox, "xyxy", 
                color=(0, 255, 0), thickness=3, label="Demo Crop"
            )
            cv2.imwrite("demo_with_box.jpg", with_box)
            print("Saved demo_with_box.jpg")
    else:
        main()
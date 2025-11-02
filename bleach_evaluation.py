import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import os
import glob

class CoralBleachingEvaluator:
    def __init__(self):
        # Load the model and processor
        self.model = ViTForImageClassification.from_pretrained("akridge/noaa-esd-coral-bleaching-vit-classifier-v1")
        self.processor = AutoImageProcessor.from_pretrained("akridge/noaa-esd-coral-bleaching-vit-classifier-v1")
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def predict_image(self, image_path):
        """Predict a single image and return the prediction"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.logits.argmax(-1).item()
        
        return prediction
    
    def evaluate_dataset(self, image_paths, true_labels):
        """Evaluate the model on a dataset"""
        predictions = []
        
        for image_path in image_paths:
            pred = self.predict_image(image_path)
            predictions.append(pred)
        
        return predictions
    
    def print_classification_report(self, true_labels, predictions):
        """Print classification report in the requested format"""
        target_names = list(self.id2label.values())
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=list(self.id2label.keys())
        )
        
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Print in the requested format
        print(f"{'precision':>12}{'recall':>12}{'f1-score':>12}")
        
        for i, label in enumerate(target_names):
            print(f"{label:<12}{precision[i]:>8.2f}{recall[i]:>8.2f}{f1[i]:>8.2f}")
        
        print(f"{'accuracy':<12}{'':<8}{'':<8}{accuracy:>8.2f}")
        print(f"{'macro avg':<12}{macro_precision:>8.2f}{macro_recall:>8.2f}{macro_f1:>8.2f}")
        print(f"{'weighted avg':<12}{weighted_precision:>8.2f}{weighted_recall:>8.2f}{weighted_f1:>8.2f}")
    
    def evaluate_from_directory(self, data_dir):
        """
        Evaluate model on images organized in subdirectories by class
        Expected structure:
        data_dir/
        ├── CORAL/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── CORAL_BL/
            ├── image3.jpg
            └── image4.jpg
        """
        all_predictions = []
        all_true_labels = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            if class_name in self.label2id:
                true_label = self.label2id[class_name]
                
                # Get all image files in the class directory
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(class_dir, ext)))
                
                for image_file in image_files:
                    pred = self.predict_image(image_file)
                    all_predictions.append(pred)
                    all_true_labels.append(true_label)
        
        if all_predictions:
            print("EVALUATION RESULTS:")
            print("="*40)
            self.print_classification_report(all_true_labels, all_predictions)
            return all_true_labels, all_predictions
        else:
            print("No dataset found. Please organize your images in class subdirectories.")
            return None, None

# Example usage
if __name__ == "__main__":
    evaluator = CoralBleachingEvaluator()
    
    # Test with single image
    image_path = "coral.jpeg"
    if os.path.exists(image_path):
        prediction = evaluator.predict_image(image_path)
        print(f"Single image prediction: {prediction} ({evaluator.id2label[prediction]})")
        print()
    
    # Example with mock data to show the format you requested
    print("EXAMPLE CLASSIFICATION REPORT:")
    print("="*40)
    
    # Mock data to demonstrate the format
    # In real usage, you would get these from your actual dataset
    true_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0] * 6  # Mix of CORAL (0) and CORAL_BL (1)
    mock_predictions = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 6  # Some correct, some wrong
    
    evaluator.print_classification_report(true_labels, mock_predictions)
    
    print("\nTo evaluate on your own dataset:")
    print("1. Organize images in folders: CORAL/ and CORAL_BL/")
    print("2. Call: evaluator.evaluate_from_directory('path/to/your/dataset')")
import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os
import glob

# ✅ Load the model and processor
model = ViTForImageClassification.from_pretrained("akridge/noaa-esd-coral-bleaching-vit-classifier-v1")
processor = AutoImageProcessor.from_pretrained("akridge/noaa-esd-coral-bleaching-vit-classifier-v1")

def predict_image(image_path):
    """Predict a single image and return the prediction"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(-1).item()
    
    return prediction

def evaluate_dataset(image_paths, true_labels):
    """Evaluate the model on a dataset and return classification report"""
    predictions = []
    
    for image_path in image_paths:
        pred = predict_image(image_path)
        predictions.append(pred)
    
    return predictions

# Get label mapping
id2label = model.config.id2label
label2id = {v: k for k, v in id2label.items()}

# For demonstration with your single image
image_path = "coral.jpeg"
prediction = predict_image(image_path)
print(f"Single image prediction: {prediction} ({id2label[prediction]})")

# Example of how to generate classification report
# You would need to provide true labels for your dataset
# For demonstration, I'll create a mock scenario

# Mock evaluation (replace this with your actual dataset)
print("\n" + "="*50)
print("CLASSIFICATION REPORT (Mock Example)")
print("="*50)

# Example true labels and predictions for demonstration
# Replace these with your actual data
true_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10  # Mock true labels
mock_predictions = [0, 1, 0, 0, 0, 1, 1, 1, 0, 1] * 10  # Mock predictions

# Generate classification report
target_names = list(id2label.values())
report = classification_report(
    true_labels, 
    mock_predictions, 
    target_names=target_names,
    digits=2
)

print(report)

# If you have a dataset directory, you can use this function:
def evaluate_from_directory(data_dir):
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
            
        if class_name in label2id:
            true_label = label2id[class_name]
            
            # Get all image files in the class directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            
            for image_file in image_files:
                pred = predict_image(image_file)
                all_predictions.append(pred)
                all_true_labels.append(true_label)
    
    if all_predictions:
        report = classification_report(
            all_true_labels, 
            all_predictions, 
            target_names=target_names,
            digits=2
        )
        print("\nEVALUATION ON YOUR DATASET:")
        print("="*40)
        print(report)
    else:
        print("No dataset found. Please organize your images in class subdirectories.")

# Uncomment the line below if you have a dataset directory
# evaluate_from_directory("path/to/your/dataset")

from ultralytics import YOLO

# Load the trained model
model = YOLO("yolov11n-cls-noaa-esd-coral-bleaching-classifier.pt")

# Predict on an image
results = model.predict(source="coral.jpeg", imgsz=224)
for result in results:
    predicted_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import BloodCancerModel
from src.dataset import preprocess_image  # If used in the script

import torch
import torchvision.transforms as transforms
from PIL import Image

# Model and Device Setup
model_path = "saved_model/blood_cancer_model_final.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = BloodCancerModel(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities.max().item()

    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        sys.exit(1)

    predicted_class, confidence = predict_image(image_path)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
class_names = {
    0: "benign",
    1: "early_pre-b",
    2: "pre-b",
    3: "pro-b"
}

if __name__ == "__main__":
    # (Rest of the script remains the same)
    predicted_class, confidence = predict_image(image_path)
    class_name = class_names.get(predicted_class, "Unknown")
    print(f"Predicted Class: {class_name}, Confidence: {confidence}")

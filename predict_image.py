import torch
from torchvision import transforms
from PIL import Image
from src.model import BloodCancerModel

def predict_image(image_path):
    # Load the model
    model_path = "saved_model/blood_cancer_model_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodCancerModel(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1).max().item()
    
    # Map class index to class name
    class_map = {0: "Benign", 1: "Early Pre-B", 2: "Pre-B", 3: "Pro-B"}
    return class_map[predicted_class.item()], confidence

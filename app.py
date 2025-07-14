from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from PIL import Image
import torch
from torchvision import transforms
from src.model import BloodCancerModel

# Initialize the Flask app
app = Flask(__name__)

# Model setup
model_path = "saved_model/blood_cancer_model_final.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BloodCancerModel(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class mapping
class_names = {0: "benign", 1: "early_pre-b", 2: "pre-b", 3: "pro-b"}

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities.max().item()

    return class_names.get(predicted_class, "Unknown"), confidence

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        predicted_class, confidence = predict_image(file_path)
        os.remove(file_path)  # Remove the uploaded file after prediction
        return render_template('result.html', predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    # Make uploads folder if doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

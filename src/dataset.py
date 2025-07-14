import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BloodCancerDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []

        # Mapping class names to indices
        self.class_to_idx = {"benign": 0, "early_pre-b": 1, "pre-b": 2, "pro-b": 3}

        # Read the dataset directory and populate paths and labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

        # Transformations for the images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

# Function for single image preprocessing, used in predict.py
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

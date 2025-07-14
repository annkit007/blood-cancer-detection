# src/blood_cancer_preprocessing.py
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the transformations for data preprocessing
def get_transform():
    transform = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.CoarseDropout(max_holes=2, max_height=16, max_width=16, p=0.5),
        ToTensorV2()
    ])
    return transform

# Custom dataset class for Albumentations-based transformations
class AlbumentationsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_idx, class_dir in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append((os.path.join(class_path, fname), class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        return img.float(), target  # Ensure the tensor is of type float

# Function to create a balanced dataset using over-sampling
class BalancedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [d for d in datasets if len(d) > 0]  # Only include non-empty datasets

    def __len__(self):
        return max([len(d) for d in self.datasets]) if self.datasets else 0

    def __getitem__(self, idx):
        dataset_idx = idx % len(self.datasets)
        dataset = self.datasets[dataset_idx]
        data_idx = idx // len(self.datasets)
        img, target = dataset[data_idx % len(dataset)]
        return img.float(), target  # Ensure the tensor is of type float

# Function to load balanced data using the transformations
def load_data(data_dir, batch_size=32):
    transform = get_transform()
    
    benign_dataset = AlbumentationsDataset(root_dir=os.path.join(data_dir, 'benign'), transform=transform)
    early_pre_b_dataset = AlbumentationsDataset(root_dir=os.path.join(data_dir, 'early_pre-b'), transform=transform)
    pre_b_dataset = AlbumentationsDataset(root_dir=os.path.join(data_dir, 'pre-b'), transform=transform)
    pro_b_dataset = AlbumentationsDataset(root_dir=os.path.join(data_dir, 'pro-b'), transform=transform)

    datasets = [benign_dataset, early_pre_b_dataset, pre_b_dataset, pro_b_dataset]
    balanced_dataset = BalancedDataset(datasets)
    
    if len(balanced_dataset) == 0:
        raise ValueError("All datasets are empty.")
    
    data_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

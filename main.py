import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import BloodCancerDataset
from src.model import BloodCancerModel
import os

# Paths
train_data_path = "split_data/train"
val_data_path = "split_data/val"
model_save_path = "saved_model/blood_cancer_model_final.pth"

# Hyperparameters
batch_size = 16
num_epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets and DataLoaders
train_dataset = BloodCancerDataset(train_data_path)
val_dataset = BloodCancerDataset(val_data_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = BloodCancerModel(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save Model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

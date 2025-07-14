import torch.nn as nn
import torchvision.models as models

class BloodCancerModel(nn.Module):
    def __init__(self, num_classes=4):
        super(BloodCancerModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

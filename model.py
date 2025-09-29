#model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights

class HybridEfficientNet(nn.Module):
    def __init__(self, num_classes, use_temporal=False):
        super().__init__()
        weights = EfficientNet_B5_Weights.DEFAULT
        base_model = models.efficientnet_b5(weights=weights)
        
        # For feature extraction
        self.backbone = nn.Sequential(*(list(base_model.children())[:-1]))

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

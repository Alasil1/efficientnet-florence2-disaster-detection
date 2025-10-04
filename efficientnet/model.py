"""model.py
Model definition extracted from the notebook.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


class DisasterClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(DisasterClassifier, self).__init__()
        self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    print("model.py: defines DisasterClassifier")

import torch
import torch.nn as nn
# Assuming these are in the same project structure
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        """
        super(VGG11Classifier, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, 3, H, W].
        Returns:
            Classification logits [B, 37].
        """
        features = self.encoder(x, return_features=False)  
        pooled_features = self.avgpool(features)
        logits = self.classifier_head(pooled_features)
        return logits
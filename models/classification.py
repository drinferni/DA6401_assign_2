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
        
        # 1. Instantiate the Encoder (Contracting Path)
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # 2. Global Average Pooling 
        # This ensures that even if input images aren't exactly 224x224, 
        # the feature map is reduced to a fixed 7x7 spatial size before flattening.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 3. Modernized VGG Classification Head
        # Standard VGG uses 4096 -> 4096 -> num_classes.
        # We include BatchNorm1d and CustomDropout as per Task 1.1.
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            
            # FC Layer 1
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            
            # FC Layer 2
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            
            # Output Layer (Logits for 37 classes)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, 3, H, W].
        Returns:
            Classification logits [B, 37].
        """
        # Extract features using the VGG11 backbone
        # We do not need skip connections for pure classification
        features = self.encoder(x, return_features=False) 
        
        # Pool to 7x7 and flatten
        pooled_features = self.avgpool(features)
        
        # Pass through the classification head
        logits = self.classifier_head(pooled_features)
        
        return logits

# Usage for Task 2.1 & 2.2:
# model = VGG11Classifier(num_classes=37, dropout_p=0.5)
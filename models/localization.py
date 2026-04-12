import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Localizer, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(512 * 7 * 7, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        features = self.encoder(x, return_features=False)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        normalized_coords = self.regression_head(pooled)
        return normalized_coords

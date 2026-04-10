import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer for single-object bounding box regression."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.
        """
        super(VGG11Localizer, self).__init__()

        # 1. Encoder (Foundational backbone from Task 1)
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # 2. Adaptive Pooling to ensure fixed input size for the head
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 3. Regression Decoder (Head)
        # We use a structure similar to the classification head but 
        # ending in 4 units for [x_center, y_center, width, height].
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(512 * 7 * 7, 512), # Reduced size is often sufficient for regression
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Output Layer: 4 continuous values
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for localization model.
        
        Returns:
            Normalized Bounding box coordinates [B, 4] 
            Format: (x_center, y_center, width, height) in range [0.0, 1.0]
        """
        # 1. Extract features using the shared encoder
        features = self.encoder(x, return_features=False)
        
        # 2. Global Average Pooling (converts [B, C, H, W] to [B, C])
        pooled = self.avgpool(features)
        # Flatten if necessary (depends on your avgpool output shape)
        pooled = torch.flatten(pooled, 1)
        
        # 3. Get normalized coordinates
        # IMPORTANT: Ensure your regression_head ends with a nn.Sigmoid() 
        # to constrain the values between 0.0 and 1.0.
        normalized_coords = self.regression_head(pooled) # [B, 4]
        
        return normalized_coords

# Note for Task 1.2: 
# In your report, you must state whether you are "freezing" the encoder weights 
# or "fine-tuning" them for this task.
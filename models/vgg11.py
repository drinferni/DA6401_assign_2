from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
# Assuming CustomDropout is in a file named custom_layers.py
from models.layers import CustomDropout 

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with skip connections for U-Net."""

    def __init__(self, in_channels: int = 3):
        super(VGG11Encoder, self).__init__()

        # Define the VGG11 configuration (number of filters per layer)
        # 'M' stands for MaxPool2d
        # VGG11: 1 conv -> M -> 1 conv -> M -> 2 conv -> M -> 2 conv -> M -> 2 conv -> M
        
        # Block 1: 64
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 256, 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 512, 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: 512, 512 (Bottleneck)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor [B, 3, H, W].
            return_features: If True, returns features from each block for U-Net skip connections.
        """
        # Feature Extraction and Skip Map Storage
        # We save features BEFORE pooling to preserve spatial resolution for the decoder
        
        f1 = self.block1(x)
        p1 = self.pool1(f1)
        
        f2 = self.block2(p1)
        p2 = self.pool2(f2)
        
        f3 = self.block3(p2)
        p3 = self.pool3(f3)
        
        f4 = self.block4(p3)
        p4 = self.pool4(f4)
        
        f5 = self.block5(p4)
        bottleneck = self.pool5(f5)

        if return_features:
            # feature_dict will be used by the U-Net style decoder for concatenation
            feature_dict = {
                "skip1": f1, # 64 channels
                "skip2": f2, # 128 channels
                "skip3": f3, # 256 channels
                "skip4": f4, # 512 channels
                "skip5": f5  # 512 channels
            }
            return bottleneck, feature_dict
        
        return bottleneck
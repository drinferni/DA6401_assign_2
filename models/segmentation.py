import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

# class for segmentation head
class UNetDecoderHead(nn.Module):

    def __init__(self, num_classes: int, dropout_p: float):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super(UNetDecoderHead, self).__init__()

        # Block_5
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self._decoder_block(512 + 512, 512, dropout_p)

        # Block_4
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(512 + 512, 512, dropout_p)

        # Block_3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(256 + 256, 256, 0.0)

        # Block_2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(128 + 128, 128, 0.0)

        # Block_1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(64 + 64, 64, 0.0)

        # Final Layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_ch, out_ch, dropout_p):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(CustomDropout(p=dropout_p))
        return nn.Sequential(*layers)

    def forward(self, bottleneck: torch.Tensor, skips: dict) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Stage 5
        x = self.up5(bottleneck)
        x = torch.cat([x, skips["skip5"]], dim=1)
        x = self.dec5(x)

        # Stage 4
        x = self.up4(x)
        x = torch.cat([x, skips["skip4"]], dim=1)
        x = self.dec4(x)

        # Stage 3
        x = self.up3(x)
        x = torch.cat([x, skips["skip3"]], dim=1)
        x = self.dec3(x)

        # Stage 2
        x = self.up2(x)
        x = torch.cat([x, skips["skip2"]], dim=1)
        x = self.dec2(x)

        # Stage 1
        x = self.up1(x)
        x = torch.cat([x, skips["skip1"]], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)

# main class
class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11UNet, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.segmentation_head = UNetDecoderHead(num_classes, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x, return_features=True)
        return self.segmentation_head(bottleneck, skips)
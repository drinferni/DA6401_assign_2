import os

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import *
import gdown

class MultiTaskPerceptionModel(nn.Module):

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super(MultiTaskPerceptionModel, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.full_classifier = VGG11Classifier()
        self.classifier_head = self.full_classifier.classifier_head
        self.full_localizer = VGG11Localizer()
        self.localizer_head = self.full_localizer.regression_head
        self.full_segmentator= VGG11UNet()
        self.segmentator_head = self.full_segmentator.segmentation_head

        try :
            gdown.download(id="1RR5KrgDdAz3HUQqtNvp-4Dw-l4whD1UE", output="classifier.pth", quiet=False)
        except:
            pass
        try:
            gdown.download(id="1heclTBH3jEAraubxbEAX_CB0ctZ-Qo-q", output="localizer.pth", quiet=False)
        except:
            pass
        try:
            gdown.download(id="1ptZWXITMOE_mac1SRZKIi9qi5FUH1qLf", output="unet.pth", quiet=False)
        except:
            pass
        self.load_from_checkpoints()

    def load_from_checkpoints(self, cls_path="classifier.pth", loc_path="localizer.pth", unet_path="unet.pth"):
        cls_ckpt = torch.load(cls_path, map_location='cpu')['state_dict']
        state_dict = cls_ckpt.get('state_dict', cls_ckpt) if isinstance(cls_ckpt, dict) else cls_ckpt
        self.load_state_dict(state_dict)
        loc_ckpt = torch.load(loc_path, map_location='cpu')['state_dict']
        state_dict = loc_ckpt.get('state_dict', loc_ckpt) if isinstance(loc_ckpt, dict) else loc_ckpt
        self.load_state_dict(state_dict)
        unet_ckpt = torch.load(unet_path, map_location='cpu')['state_dict']
        state_dict = unet_ckpt.get('state_dict', unet_ckpt) if isinstance(unet_ckpt, dict) else unet_ckpt
        self.load_state_dict(state_dict)

        # if hasattr(self, 'encoder'):
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Unified Forward Pass."""
        bottleneck, skips = self.encoder(x, return_features=True)
        pooled = self.avgpool(bottleneck)
        pooled = torch.flatten(pooled, 1)
        class_logits = self.classifier_head(pooled)
        loc_coords = self.localizer_head(pooled) 
        seg_logits = self.segmentator_head(bottleneck,skips)

        return {
            'classification': class_logits,
            'localization': loc_coords,  # Now returns normalized [0, 1]
            'segmentation': seg_logits
        }
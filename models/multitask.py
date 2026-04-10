import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import *
import gdown

class MultiTaskPerceptionModel(nn.Module):
    """Unified shared-backbone multi-task model for the Oxford-IIIT Pet Pipeline."""

    # classifier.pth, localizer.pth, unet.pth

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super(MultiTaskPerceptionModel, self).__init__()

        # 1. SHARED BACKBONE (Contracting Path)
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # 2. CLASSIFICATION BRANCH
        # We reuse the head structure from Task 1.1
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.full_classifier = VGG11Classifier()
        self.classifier_head = self.full_classifier.classifier_head

        # 3. LOCALIZATION BRANCH
        self.full_localizer = VGG11Localizer()
        self.localizer_head = self.full_localizer.regression_head
        

        self.full_segmentator= VGG11UNet()
        self.segmentator_head = self.full_segmentator.segmentation_head

        gdown.download(id="1C73aVXsmAdoVlq5URN6GeOl_k8QbGl9-", output="classifier.pth", quiet=False)
        gdown.download(id="1_j9NO2fW52XH2bPCB1IU-Dq0Ec-SeDfp", output="localizer.pth", quiet=False)
        gdown.download(id="1FnxEYU5SRtx81BTw_1zS5GcXRJGFDMcQ", output="unet.pth", quiet=False)

    def load_from_checkpoints(self, cls_path = "classifier.pth", loc_path = "localizer.pth", unet_path = "unet.pth"):
        # --- Load Task 1 (Backbone + Classifier Head) ---
        cls_ckpt = torch.load(cls_path, map_location='cpu')['state_dict']
        # This will fill both your backbone and your classifier head
        self.load_state_dict(cls_ckpt, strict=False)
        print("Loaded backbone and classifier head from Task 1")

        # --- Load Task 2 (Localizer Head only) ---
        loc_ckpt = torch.load(loc_path, map_location='cpu')['state_dict']
        # Extract only the keys that belong to the localizer head
        loc_head_weights = {k: v for k, v in loc_ckpt.items() if "localizer_head" in k}
        self.load_state_dict(loc_head_weights, strict=False)
        print("Loaded localizer head from Task 2")

        # --- Load Task 3 (Segmentation Decoder only) ---
        unet_ckpt = torch.load(unet_path, map_location='cpu')['state_dict']
        # Extract only the keys that belong to the UNet decoder/head
        seg_head_weights = {k: v for k, v in unet_ckpt.items() if "segmentation_decoder" in k}
        self.load_state_dict(seg_head_weights, strict=False)
        print("Loaded segmentation head from Task 3")

    def forward(self, x: torch.Tensor):
        """Unified Forward Pass."""
        # Step 1: Shared Encoder Pass (Contracting Path)
        # Get bottleneck features and skip connections for U-Net
        bottleneck, skips = self.encoder(x, return_features=True)
        
        # Step 2: Shared Bottleneck Pooling
        # Ensure pooled is flattened [B, C] if heads are Linear layers
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
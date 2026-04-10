import os

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

        try :
            gdown.download(id="1HqClAACl_sdaiK59YmA0HEqiQiPoIUWLeJ_", output="classifier.pth", quiet=False)
        except:
            pass
        try:
            gdown.download(id="1yK0Lk8zhdsaNrrHhrEgkSp44LlXpWE_JVhu", output="localizer.pth", quiet=False)
        except:
            pass
        try:
            gdown.download(id="1BtB5vlldfX45j387UYStMrXb-54NLY2Y9g", output="unet.pth", quiet=False)
        except:
            pass
        try:
            gdown.download(id="10_5WQgWklSZH_6085aikf1eBN0iRpHGf", output="multi.pth", quiet=False)
        except:
            pass

    def load_from_checkpoints(self, cls_path="classifier.pth", loc_path="localizer.pth", unet_path="unet.pth", final_path="multi.pth"):
        # --- STRATEGY 1: Load Consolidated "Final" Checkpoint ---
        if os.path.exists(final_path):
            print(f"📦 Found consolidated checkpoint: {final_path}. Loading entire model...")
            ckpt = torch.load(final_path, map_location='cpu')
            # Handle cases where the dict is wrapped in ['state_dict']
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            self.load_state_dict(state_dict)
            print("✅ Entire model loaded from final checkpoint.")
            
        # --- STRATEGY 2: Load Individual Task Checkpoints ---
        else:
            print(f"🔍 {final_path} not found. Loading from individual task weights...")
            
            # Check if all individual files exist before starting
            for p in [cls_path, loc_path, unet_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"CRITICAL: Weight file '{p}' not found. Cannot assemble model.")

            # 1. Load Task 1: Classification + Shared Backbone
            cls_ckpt = torch.load(cls_path, map_location='cpu')['state_dict']
            self.load_state_dict(cls_ckpt, strict=False)
            print("✅ Backbone and Classifier loaded.")

            # 2. Load Task 2: Localization
            loc_ckpt = torch.load(loc_path, map_location='cpu')['state_dict']
            loc_mapped = {k.replace("regression_head", "localizer_head"): v 
                            for k, v in loc_ckpt.items() if "regression_head" in k}
            self.load_state_dict(loc_mapped, strict=False)
            print(f"✅ Localization head loaded ({len(loc_mapped)} layers).")

            # 3. Load Task 3: Segmentation
            unet_ckpt = torch.load(unet_path, map_location='cpu')['state_dict']
            seg_mapped = {k.replace("segmentation_head", "segmentator_head"): v 
                            for k, v in unet_ckpt.items() if "segmentation_head" in k}
            self.load_state_dict(seg_mapped, strict=False)
            print(f"✅ Segmentation head loaded ({len(seg_mapped)} layers).")

        # --- FINAL SETUP (Applies to both strategies) ---
        # 4. CRITICAL: Freeze the backbone to protect Macro-F1
        if hasattr(self, 'encoder'):
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("❄️ Encoder Frozen.")

        self.eval() 
        print("🚀 Model set to Eval mode. Loading sequence complete.")


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
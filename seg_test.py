import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import gc

# Import your classes
from data.pets_dataset import OxfordIIITPetDataset
from models import *
from losses.iou_loss import IoULoss
import torch.nn.functional as F


def visualize_prediction(model, input_tensor, num_classes=3):
    """
    input_tensor: A tensor of shape [1, 3, 224, 224]
    """
    model.eval() # 1. Set to evaluation mode (turns off dropout/batchnorm)
    
    with torch.no_grad():
        # 2. Get the model output (logits)
        logits = model(input_tensor) # Shape: [1, num_classes, 224, 224]
        
        # 3. Convert logits to probabilities (optional, for heatmaps)
        probs = torch.softmax(logits, dim=1)
        
        # 4. Get the predicted class per pixel
        # [1, 3, 224, 224] -> [224, 224]
        prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # 5. Prepare input image for display
    # Convert from [1, 3, 224, 224] to [224, 224, 3]
    input_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Un-normalize the image if you applied standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = std * input_img + mean
    input_img = np.clip(input_img, 0, 1)

    # 6. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(input_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Use a colormap like 'jet', 'viridis', or 'tab10' for classes
    im = ax[1].imshow(prediction, cmap='tab10', vmin=0, vmax=num_classes-1)
    ax[1].set_title("Predicted Segmentation")
    ax[1].axis("off")
    
    plt.colorbar(im, ax=ax[1], ticks=range(num_classes), label="Class ID")
    plt.show()

# --- Example Usage ---
# Assuming 'model' is your VGG11UNet and 'img_tensor' is your input

def get_dataloader():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    train_ds = OxfordIIITPetDataset(root_dir="./data", split="trainval", transform=train_transform)
    return DataLoader(train_ds, batch_size=1, shuffle=True)

train_loader = get_dataloader()

for epoch in range(1):
    total_loss = 0
    for images, targets in tqdm(train_loader):
        visualize_prediction(VGG11UNet().to('cuda'), images)
        break
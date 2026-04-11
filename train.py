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


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: Raw logits from the model (before sigmoid)
            target: Ground truth masks (same shape as pred)
        """
        # 1. Binary Cross Entropy with Logits
        # This is more numerically stable than applying sigmoid then BCE
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # 2. Dice Loss
        pred_sig = torch.sigmoid(pred)
        
        # Flatten tensors to ensure the sum works correctly across batches and channels
        intersection = (pred_sig * target).sum()
        dice_coeff = (2. * intersection) / (pred_sig.sum() + target.sum() + 1e-6)
        dice_loss = 1 - dice_coeff
        
        return bce + dice_loss


# --- Helper for Memory Cleanup ---
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# --- Common Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 16

def get_dataloader():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    train_ds = OxfordIIITPetDataset(root_dir="./data", split="trainval", transform=train_transform)
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

def save_checkpoint(model, epoch, metric, filename):
    """Saves checkpoint in the requested format."""
    payload = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": metric,
    }
    torch.save(payload, filename)
    print(f"Checkpoint saved to {filename} (Metric: {metric:.4f})")

# --- Training Functions ---

def train_classifier(train_loader):
    print("\n--- Training Classifier ---")
    model = VGG11Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Cls Epoch {epoch}"):
            images, labels = images.to(DEVICE), targets['label'].to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"cls_batch_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        wandb.log({"cls_epoch_loss": avg_loss})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, epoch, best_loss, "classifier.pth")

    del model, optimizer
    cleanup()

def train_localizer(train_loader):
    print("\n--- Training Localizer ---")
    model = VGG11Localizer().to(DEVICE)
    criterion_iou = IoULoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Loc Epoch {epoch}"):
            images, boxes = images.to(DEVICE), targets['bbox'].to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_iou(output, boxes) + criterion_mse(output, boxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"loc_batch_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        wandb.log({"loc_epoch_loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, epoch, best_loss, "localizer.pth")

    del model, optimizer
    cleanup()

def train_segmenter(train_loader):
    print("\n--- Training Segmenter ---")
    model = VGG11UNet().to(DEVICE)
    criterion = seg_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Seg Epoch {epoch}"):
            images = images.to(DEVICE)
            masks = targets['mask'].to(DEVICE).long().squeeze(1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"seg_batch_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        wandb.log({"seg_epoch_loss": avg_loss})


        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, epoch, best_loss, "unet.pth")

    del model, optimizer
    cleanup()

def train_multi(train_loader):
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3).to(DEVICE)
    
    # Example: Partial Fine-Tuning strategy
    # for param in model.encoder.block1.parameters(): param.requires_grad = False

    # 5. Losses & Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_loc = IoULoss()
    criterion_iou = IoULoss()
    criterion_mse = nn.MSELoss() # Assuming standard PyTorch MSE
    weights = torch.tensor([0.1, 1.0, 1.0]).cuda() 
    criterion_seg = nn.CrossEntropyLoss(weight=weights) # For 3-class trimap
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float('inf')

    # 6. Loop
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, targets in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = targets['label'].to(DEVICE)
            boxes = targets['bbox'].to(DEVICE)
            masks = targets['mask'].to(DEVICE).long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined Loss Calculation (Task 1.4)
            loss_cls = criterion_cls(outputs['classification'], labels)
            # loss_loc = criterion_loc(outputs['localization'], boxes)
            # Calculate individual losses
            loss_iou = criterion_iou(outputs['localization'], boxes)
            loss_mse = criterion_mse(outputs['localization'], boxes)

            # Combine them
            loss_loc = loss_iou + loss_mse
            loss_seg = criterion_seg(outputs['segmentation'], masks)
            
            # Weighing losses (adjust based on empirical results)
            loss = 0.01* loss_cls +  0.01 * loss_loc + 20.0 * loss_seg
            # print(loss_cls.item(), loss_loc.item(), loss_seg.item())
            loss.backward()
            optimizer.step()

            wandb.log({
                "epoch": epoch,
                "cls_loss": loss_cls.item(),
                "loc_loss": loss_loc.item(),
                "seg_loss": loss_seg.item()
            })
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
        })
        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, epoch, best_loss, "multi.pth")

# --- Main Pipeline ---
def main():
    wandb.init(project="DA6401-Assignment2", name="sequential-split-models")
    
    train_loader = get_dataloader()

    # Train models one by one to save GPU space
    # train_classifier(train_loader)
    # train_localizer(train_loader)
    # train_segmenter(train_loader)    
    train_multi(train_loader)

    print("\nAll models trained sequentially.")
    wandb.finish()

if __name__ == "__main__":
    main()
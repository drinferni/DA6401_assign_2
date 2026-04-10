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
from train import cleanup, save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
BATCH_SIZE = 16

def get_dataloader():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    train_ds = OxfordIIITPetDataset(root_dir="./data", split="test", transform=train_transform)
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


def train_multi(test_loader):
    print("\n--- Training Multi ---")

    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    model.load_from_checkpoints()
    model.to(DEVICE).eval()

    
    criterion_cls = nn.CrossEntropyLoss()
    # criterion_loc = IoULoss()
    criterion_iou = IoULoss()
    criterion_mse = nn.MSELoss() # Assuming standard PyTorch MSE
    criterion_seg = nn.CrossEntropyLoss() # For 3-class trimap
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6. Loop
    # model.train()
    # for epoch in range(2):
    #     total_loss = 0
    #     for images, targets in tqdm(train_loader):
    #         images = images.to(DEVICE)
    #         labels = targets['label'].to(DEVICE)
    #         boxes = targets['bbox'].to(DEVICE)
    #         masks = targets['mask'].to(DEVICE).long().squeeze(1)

    #         optimizer.zero_grad()
    #         outputs = model(images)

    #         # Combined Loss Calculation (Task 1.4)
    #         loss_cls = criterion_cls(outputs['classification'], labels)
    #         # loss_loc = criterion_loc(outputs['localization'], boxes)
    #         # Calculate individual losses
    #         loss_iou = criterion_iou(outputs['localization'], boxes)
    #         loss_mse = criterion_mse(outputs['localization'], boxes)

    #         # Combine them
    #         loss_loc = loss_iou + loss_mse
    #         loss_seg = criterion_seg(outputs['segmentation'], masks)
            
    #         # Weighing losses (adjust based on empirical results)
    #         loss = loss_cls + (2.0 * loss_loc) + loss_seg
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()

    #     # Log to W&B
    #     wandb.log({
    #         "epoch": epoch,
    #         "train_loss": total_loss / len(train_loader),
    #         "cls_loss": loss_cls.item(),
    #         "loc_loss": loss_loc.item(),
    #         "seg_loss": loss_seg.item()
    #     })
    #     print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    
    total_test_loss = 0
    total_cls_loss = 0
    total_loc_loss = 0
    total_seg_loss = 0
    
    # Optional: Track metrics for better testing insights
    correct_cls = 0
    total_samples = 0

    # 2. Disable gradient tracking
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            labels = targets['label'].to(DEVICE)
            boxes = targets['bbox'].to(DEVICE)
            masks = targets['mask'].to(DEVICE).long().squeeze(1)

            # 3. Forward pass
            outputs = model(images)

            # Calculate individual losses (same logic as training for comparison)
            loss_cls = criterion_cls(outputs['classification'], labels)
            
            loss_iou = criterion_iou(outputs['localization'], boxes)
            loss_mse = criterion_mse(outputs['localization'], boxes)
            loss_loc = loss_iou + loss_mse
            
            loss_seg = criterion_seg(outputs['segmentation'], masks)
            
            # Combine losses
            loss = loss_cls + loss_loc + loss_seg

            # Accumulate losses
            total_test_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_loc_loss += loss_loc.item()
            total_seg_loss += loss_seg.item()

            # --- Calculate Additional Metrics (Recommended) ---
            # Classification Accuracy
            _, predicted = torch.max(outputs['classification'], 1)
            total_samples += labels.size(0)
            correct_cls += (predicted == labels).sum().item()
            
            # You could also add Mask IoU or Bbox IoU metrics here
            # --------------------------------------------------

    # Average metrics
    avg_loss = total_test_loss / len(test_loader)
    accuracy = 100 * correct_cls / total_samples

    # Log to W&B (use "test/" prefix to separate from "train/")
    wandb.log({
        "test_loss": avg_loss,
        "test_cls_loss": total_cls_loss / len(test_loader),
        "test_loc_loss": total_loc_loss / len(test_loader),
        "test_seg_loss": total_seg_loss / len(test_loader),
        "test_accuracy": accuracy
    })

    print(f"\n[Test Results] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

def main():
    wandb.init(project="DA6401-Assignment2", name="sequential-split-models")
    
    train_loader = get_dataloader()

    # Train models one by one to save GPU space
    train_multi(train_loader)

    print("\nAll models trained sequentially.")
    wandb.finish()

if __name__ == "__main__":
    main()
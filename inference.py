import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# Import your classes
from data.pets_dataset import OxfordIIITPetDataset
from models import *
from losses.iou_loss import IoULoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

def calculate_bbox_iou(pred_box, gt_box):
    """Calculates IoU for [x_center, y_center, w, h] format."""
    # Convert center to corners
    p_x1 = pred_box[:, 0] - pred_box[:, 2]/2
    p_y1 = pred_box[:, 1] - pred_box[:, 3]/2
    p_x2 = pred_box[:, 0] + pred_box[:, 2]/2
    p_y2 = pred_box[:, 1] + pred_box[:, 3]/2

    g_x1 = gt_box[:, 0] - gt_box[:, 2]/2
    g_y1 = gt_box[:, 1] - gt_box[:, 3]/2
    g_x2 = gt_box[:, 0] + gt_box[:, 2]/2
    g_y2 = gt_box[:, 1] + gt_box[:, 3]/2

    # Intersection
    i_x1 = torch.max(p_x1, g_x1)
    i_y1 = torch.max(p_y1, g_y1)
    i_x2 = torch.min(p_x2, g_x2)
    i_y2 = torch.min(p_y2, g_y2)

    inter_area = torch.clamp(i_x2 - i_x1, min=0) * torch.clamp(i_y2 - i_y1, min=0)
    
    # Union
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    g_area = (g_x2 - g_x1) * (g_y2 - g_y1)
    union_area = p_area + g_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def calculate_dice(pred_mask, gt_mask, num_classes=3):
    """Calculates Macro-Dice Score."""
    dice_scores = []
    # pred_mask: [B, H, W], gt_mask: [B, H, W]
    for i in range(num_classes):
        p = (pred_mask == i).float()
        g = (gt_mask == i).float()
        intersection = (p * g).sum()
        dice = (2. * intersection) / (p.sum() + g.sum() + 1e-6)
        dice_scores.append(dice.item())
    return np.mean(dice_scores)

def decode_segmap(image, num_classes=3):
    # Define a color map for each class (RGB)
    label_colors = np.array([
        (0, 0, 0),       # Class 0: Background
        (255, 0, 0),     # Class 1: Red
        (0, 0, 255),     # Class 2: Blue
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Usage inside the plotting section:
# colored_mask = decode_segmap(prediction)
# ax[1].imshow(colored_mask)

def train_multi(test_loader):
    print("\n--- Evaluating Multi-Task Pipeline ---")

    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    model.load_from_checkpoints()
    model.to(DEVICE)
    model.eval()

    # Metrics Accumulators
    all_cls_preds = []
    all_cls_gt = []
    all_ious = []
    all_dices = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            labels = targets['label'].to(DEVICE)
            boxes = targets['bbox'].to(DEVICE)
            masks = targets['mask'].to(DEVICE).long().squeeze(1)

            outputs = model(images)

            # print(outputs['segmentation'][0])
            # return
            # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            # ax[1].imshow(c)
            # plt.show()


            # 1. Classification Data
            _, preds = torch.max(outputs['classification'], 1)
            all_cls_preds.extend(preds.cpu().numpy())
            all_cls_gt.extend(labels.cpu().numpy())

            # 2. Localization IoU
            # Note: If your loss is 17000, your boxes are likely in pixels (0-224)
            # but your model might be outputting 0-1. Ensure they match here!
            batch_ious = calculate_bbox_iou(outputs['localization'], boxes)
            all_ious.extend(batch_ious.cpu().numpy())

            # 3. Segmentation Dice
            seg_preds = torch.argmax(outputs['segmentation'], dim=1)
            print(seg_preds[0])
            batch_dice = calculate_dice(seg_preds, masks)
            all_dices.append(batch_dice)

    # --- FINAL METRIC CALCULATION ---
    
    # Macro-F1
    macro_f1 = f1_score(all_cls_gt, all_cls_preds, average='macro')

    # Acc@IoU
    ious_np = np.array(all_ious)
    acc_iou_50 = np.mean(ious_np >= 0.5) * 100
    acc_iou_75 = np.mean(ious_np >= 0.75) * 100

    # Macro-Dice
    avg_dice = np.mean(all_dices)

    print(f"\n[Pipeline Metrics]")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Acc@IoU=0.5: {acc_iou_50:.2f}%")
    print(f"Acc@IoU=0.75: {acc_iou_75:.2f}%")
    print(f"Macro-Dice: {avg_dice:.4f}")

    # Log to W&B
    wandb.log({
        "Macro-F1": macro_f1,
        "Acc@IoU=0.5": acc_iou_50,
        "Acc@IoU=0.75": acc_iou_75,
        "Macro-Dice": avg_dice,
        "test_accuracy": (np.array(all_cls_preds) == np.array(all_cls_gt)).mean() * 100
    })

    # Error Check for your threshold
    if macro_f1 < 0.3:
        print(f"✘ CLASSIFICATION F1 < 0.3: Macro-F1 = {macro_f1:.4f}")



def get_dataloader():
    # Use standard validation transforms (No random flipping for test)
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    # NOTE: Using split="test" as per your code
    test_ds = OxfordIIITPetDataset(root_dir="./data", split="trainval", transform=test_transform)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

def main():
    wandb.init(project="DA6401-Assignment2", name="final-metric-eval")
    test_loader = get_dataloader()
    train_multi(test_loader)
    wandb.finish()

if __name__ == "__main__":
    main()
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
    p_x1 = pred_box[:, 0] - pred_box[:, 2]/2
    p_y1 = pred_box[:, 1] - pred_box[:, 3]/2
    p_x2 = pred_box[:, 0] + pred_box[:, 2]/2
    p_y2 = pred_box[:, 1] + pred_box[:, 3]/2

    g_x1 = gt_box[:, 0] - gt_box[:, 2]/2
    g_y1 = gt_box[:, 1] - gt_box[:, 3]/2
    g_x2 = gt_box[:, 0] + gt_box[:, 2]/2
    g_y2 = gt_box[:, 1] + gt_box[:, 3]/2

    i_x1 = torch.max(p_x1, g_x1)
    i_y1 = torch.max(p_y1, g_y1)
    i_x2 = torch.min(p_x2, g_x2)
    i_y2 = torch.min(p_y2, g_y2)

    inter_area = torch.clamp(i_x2 - i_x1, min=0) * torch.clamp(i_y2 - i_y1, min=0)
    
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    g_area = (g_x2 - g_x1) * (g_y2 - g_y1)
    union_area = p_area + g_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def calculate_dice(pred_mask, gt_mask, num_classes=3):
    dice_scores = []
    for i in range(num_classes):
        p = (pred_mask == i).float()
        g = (gt_mask == i).float()
        intersection = (p * g).sum()
        dice = (2. * intersection) / (p.sum() + g.sum() + 1e-6)
        dice_scores.append(dice.item())
    return np.mean(dice_scores)


# funciton to test multi task pipeline
def train_multi(test_loader):

    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
    model.load_from_checkpoints()
    model.to(DEVICE)
    model.eval()

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

            _, preds = torch.max(outputs['classification'], 1)
            all_cls_preds.extend(preds.cpu().numpy())
            all_cls_gt.extend(labels.cpu().numpy())

            batch_ious = calculate_bbox_iou(outputs['localization'], boxes)
            all_ious.extend(batch_ious.cpu().numpy())

            seg_preds = torch.argmax(outputs['segmentation'], dim=1)
            print(seg_preds[0])
            batch_dice = calculate_dice(seg_preds, masks)
            all_dices.append(batch_dice)


    # calculating metric as in tests
    macro_f1 = f1_score(all_cls_gt, all_cls_preds, average='macro')
    ious_np = np.array(all_ious)
    acc_iou_50 = np.mean(ious_np >= 0.5) * 100
    acc_iou_75 = np.mean(ious_np >= 0.75) * 100
    avg_dice = np.mean(all_dices)

    print(f"\n[Pipeline Metrics]")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Acc@IoU=0.5: {acc_iou_50:.2f}%")
    print(f"Acc@IoU=0.75: {acc_iou_75:.2f}%")
    print(f"Macro-Dice: {avg_dice:.4f}")

    wandb.log({
        "Macro-F1": macro_f1,
        "Acc@IoU=0.5": acc_iou_50,
        "Acc@IoU=0.75": acc_iou_75,
        "Macro-Dice": avg_dice,
        "test_accuracy": (np.array(all_cls_preds) == np.array(all_cls_gt)).mean() * 100
    })

def get_dataloader():
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    test_ds = OxfordIIITPetDataset(root_dir="./data", split="trainval", transform=test_transform)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

def main():
    wandb.init(project="DA6401-Assignment2", name="final-metric-eval")
    test_loader = get_dataloader()
    train_multi(test_loader)
    wandb.finish()

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) loss for bounding box regression.
    Mathematically: Loss = 1 - (Intersection / Union)
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction: {reduction}. Expected 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss between predicted and target bounding boxes.
        
        Args:
            pred_boxes: [B, 4] in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] in (x_center, y_center, width, height) format.
            
        Returns:
            Scalar tensor (if mean/sum) or [B] tensor (if none).
        """
        # 1. Convert from center format [cx, cy, w, h] to corner format [x1, y1, x2, y2]
        # x1, y1 = top-left; x2, y2 = bottom-right
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # 2. Determine the coordinates of the intersection rectangle
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        # 3. Calculate Intersection Area
        # clamp(min=0) ensures that if boxes don't overlap, area is 0 (not negative)
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        # 4. Calculate Union Area
        # Union = Area1 + Area2 - Intersection
        area_pred = pred_boxes[:, 2] * pred_boxes[:, 3]
        area_target = target_boxes[:, 2] * target_boxes[:, 3]
        union = area_pred + area_target - intersection

        # 5. Calculate IoU and Loss
        # eps prevents division by zero if union is 0
        iou = intersection / (union + self.eps)
        loss = 1 - iou

        # 6. Apply Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
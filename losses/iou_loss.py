import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
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
        p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # calculating the intersection
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        # Calculating Intersection Area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_w * inter_h

        # Calculating Union Area
        area_pred = pred_boxes[:, 2] * pred_boxes[:, 3]
        area_target = target_boxes[:, 2] * target_boxes[:, 3]
        union = area_pred + area_target - intersection

        # calculating IoU and Loss
        # eps to prevent division by 0 error as a corner case
        iou = intersection / (union + self.eps)
        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
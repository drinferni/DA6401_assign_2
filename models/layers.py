import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """
    Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        self.prob = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, N].

        Returns:
            Output tensor with dropout applied during training.
        """
        if not self.training or self.prob == 0:
            return x
        if self.prob == 1:
            return torch.zeros_like(x)
        mask = (torch.rand_like(x) > self.prob).float()
        scale = 1.0 / (1.0 - self.prob)
        return x * mask * scale
import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """
    Custom Dropout layer implementing Inverted Dropout.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Probability of an element to be zeroed. Default: 0.5
        """
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, N].

        Returns:
            Output tensor with dropout applied during training.
        """
        # 1. Check if the model is in training mode.
        # Dropout should only be applied during training.
        if not self.training or self.p == 0:
            return x
        
        # Handling the edge case where p is 1.0 (all units dropped)
        if self.p == 1:
            return torch.zeros_like(x)

        # 2. Generate the binary mask.
        # torch.rand_like generates values between [0, 1).
        # Elements > p become 1 (kept), elements <= p become 0 (dropped).
        # This ensures that exactly (1-p) fraction of elements are kept on average.
        mask = (torch.rand_like(x) > self.p).float()

        # 3. Apply Inverted Dropout Scaling.
        # To maintain the same expected value for activations during inference,
        # we scale the surviving units by 1 / (1 - p) during training.
        scale = 1.0 / (1.0 - self.p)
        
        return x * mask * scale
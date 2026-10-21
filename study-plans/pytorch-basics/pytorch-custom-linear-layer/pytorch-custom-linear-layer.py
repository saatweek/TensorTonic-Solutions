import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.randn(size=(out_features, in_features), requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(out_features, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        return x@self.weight.T + self.bias

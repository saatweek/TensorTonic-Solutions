import torch
import torch.nn.functional as F

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == "relu":
        return F.relu(x)
    elif method == "sigmoid":
        return torch.sigmoid(x)
    elif method == "tanh":
        return torch.tanh(x)
    elif method == "leaky_relu":
        return F.leaky_relu(x, negative_slope=0.01)
    else:
        raise ValueError(f"Unknown operation: {method}")

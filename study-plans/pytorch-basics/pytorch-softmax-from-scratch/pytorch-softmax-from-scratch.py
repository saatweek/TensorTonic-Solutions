import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    return torch.nn.functional.softmax(logits)

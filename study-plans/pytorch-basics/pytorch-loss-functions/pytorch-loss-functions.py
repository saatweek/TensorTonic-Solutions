import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    method = method.upper()
    if method == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif method == "CROSS_ENTROPY":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif method == "HUBER":
        loss_fn = torch.nn.HuberLoss(delta=delta)
    else:
        raise ValueError(f"Unknown Method : {method}")

    return loss_fn(pred, target).item()
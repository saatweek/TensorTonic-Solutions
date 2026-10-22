import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mu = X.mean(axis=0)
    var = ((X-mu)**2).mean(axis=0)
    # var = X.var(dim=0, unbiased=False) ## does the same thing as above
    return (gamma * (X-mu)/torch.sqrt(var+eps)) + beta
    #or you can execute the following lines below
    # norm = (X-mu) / torch.sqrt(var + eps)
    # return gamma * norm + beta
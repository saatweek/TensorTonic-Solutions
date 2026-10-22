import torch
import torch.nn.functional as F

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int,) -> tuple[torch.Tensor, torch.Tensor]:

    if accum_steps < 1:
        raise ValueError("accum_steps must be positive")
    if not micro_batches:
        raise ValueError("micro_batches must not be empty")

    w = torch.nn.Parameter(w_init.detach().clone().float())
    optimizer = torch.optim.SGD([w], lr=lr)
    last_mean_gradient = None

    for start in range(0, len(micro_batches), accum_steps):
        group = micro_batches[start:start + accum_steps]
        optimizer.zero_grad(set_to_none=True)

        for x, target in group:
            x = x.to(device=w.device, dtype=w.dtype)
            target = torch.as_tensor(target, device=w.device, dtype=w.dtype)
            pred = torch.dot(w, x)
            loss = F.mse_loss(pred, target)

            # Accumulate the mean gradient for this group.
            (loss / len(group)).backward()

        last_mean_gradient = w.grad.detach().clone()
        optimizer.step()

    return w.detach(), last_mean_gradient
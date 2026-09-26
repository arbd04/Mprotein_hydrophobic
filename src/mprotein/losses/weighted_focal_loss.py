import torch
import torch.nn.functional as F


def weighted_focal_loss(x_logit, y, weight, alpha=None, gamma=1.0, reduction="mean"):
    bce = F.binary_cross_entropy_with_logits(x_logit, y, reduction="none")
    p_t = torch.exp(-bce)
    if alpha != None:
        alpha = torch.as_tensor(alpha, device=x_logit.device, dtype=x_logit.dtype)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        loss = alpha_t * (1 - p_t) ** gamma * bce * weight.to(x_logit.device)
    else:
        loss = (1 - p_t) ** gamma * bce * weight.to(x_logit.device)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()

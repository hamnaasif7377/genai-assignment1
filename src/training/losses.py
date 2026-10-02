"""Loss functions for Task 1 (and shared by Tasks 2/3 specialists)."""
import torch.nn.functional as F
from pytorch_msssim import ssim


def combined_loss(pred, target, alpha=0.8):
    """L1 + (1 - SSIM), weighted by alpha. Returns (total_loss, l1_value, ssim_value)."""
    l1 = F.l1_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    total = alpha * l1 + (1 - alpha) * (1 - ssim_val)
    return total, l1.item(), ssim_val.item()

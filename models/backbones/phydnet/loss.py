import torch as torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

class L1L2(torch.nn.Module):
    def __init__(self):
        super(L1L2, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, a, b):
        return self.L1(a, b) + self.L2(a, b)

class ICLoss(nn.Module):
    """Intensity Classification Loss.

    Performs binary CrossEntropyLoss weighted towards high intensities (1)
    on thresholded images. Expects input of shape [B, S, C, H, W] or [B, C, H, W]

    Attributes:
        alpha: A float weight multiplier for the 1 class.
        beta: A float threshold for intensity.
    """
    def __init__(self, alpha: float = 5.,
                 beta: float = 0.67) -> None:
        super().__init__()

        self.ce = CrossEntropyLoss(weight=torch.tensor([1., alpha]))
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        # threshold target intensities & remove C dim -> [B, S, H, W] / [B, H, W]
        t = (target > self.beta).long()[..., 0, :, :]

        # compute probabilites for 0 class -> [B, C, S, H, W] / [B, C, H, W]
        p = torch.concat([1 - output, output], dim=-3).transpose(-3, 1)

        return self.ce(p, t)

def module_norm(model):
    """Computes Frobenius norm of all the parameters of a model combined.
    """
    _norm = 0.
    for p in model.parameters():
        _norm += torch.sum(p**2)
    
    return torch.sqrt(_norm)
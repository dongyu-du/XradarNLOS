import torch
import torch.nn as nn


class HeatmapFocalLoss(nn.Module):
    """CenterNet-style focal loss for heatmap regression."""

    def __init__(self, alpha: int = 2, beta: int = 4):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred  = torch.clamp(pred.sigmoid(), 1e-4, 1 - 1e-4)
        pos   = gt.eq(1).float()
        neg   = gt.lt(1).float()
        neg_w = torch.pow(1 - gt, self.beta)
        pos_loss = torch.log(pred)       * torch.pow(1 - pred, self.alpha) * pos
        neg_loss = torch.log(1 - pred)   * torch.pow(pred,     self.alpha) * neg_w * neg
        n_pos = pos.sum()
        if n_pos == 0:
            return -neg_loss.sum()
        return -(pos_loss.sum() + neg_loss.sum()) / n_pos

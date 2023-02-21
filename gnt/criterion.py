import torch.nn as nn
from utils import img2mse
import torch

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log, pred_mask):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        # if "mask" in outputs:
        #     pred_mask = outputs["mask"].float()
        # else:
        #     pred_mask = None
        if pred_mask:
            mask = torch.exp(-torch.abs(pred_rgb - ray_batch["rgb"])).sum(dim=-1)
        else:
            mask = None

        gt_rgb = ray_batch["rgb"]

        loss = img2mse(pred_rgb, gt_rgb, mask)

        return loss, scalars_to_log

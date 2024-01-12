import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MPJPE(nn.Module):
    def __init__(self):
        super(MPJPE, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):
        # Calculate the normal
        normal = torch.norm(pre_xyz - gt_xyz, dim=2) # Calculate along the third dimension

        # Select valid key points through mask
        masked_normal = torch.masked_select(normal, keypoint_vis.squeeze(-1).to(dtype=torch.bool))

        # If there are no valid keypoints, return 0 to avoid division by zero
        if masked_normal.numel() == 0:
            return torch.tensor(0.0, device=normal.device)

        # Calculate the average to get the overall mpjpe
        avg_mpjpe = torch.mean(masked_normal)

        return avg_mpjpe
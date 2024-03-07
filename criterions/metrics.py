import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MPJPE(nn.Module):
    def __init__(self):
        super(MPJPE, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):

        # Calculate the Euclidean distance (without squaring)
        distance = torch.sqrt(torch.sum((pre_xyz - gt_xyz) ** 2, dim=2))  # Calculate along the third dimension

        # Apply the mask to select valid keypoints
        # Convert keypoint_vis to bool for masking
        masked_distance = torch.masked_select(distance, keypoint_vis.squeeze(-1).to(dtype=torch.bool))

        # If there are no valid keypoints, return 0 to avoid division by zero
        if masked_distance.numel() == 0:
            return torch.tensor(0.0, device=distance.device)

        # Calculate the mean to get the overall MPJPE
        avg_mpjpe = torch.mean(masked_distance)
        avg_mpjpe *= 1000

        return avg_mpjpe
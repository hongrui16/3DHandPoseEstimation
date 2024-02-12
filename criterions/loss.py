import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):
        # Calculate the square of the Euclidean distance
        squared_distance = torch.sum((pre_xyz - gt_xyz) ** 2, dim=2)  # Calculate along the third dimension

        # Apply the mask to select valid keypoints
        # Convert keypoint_vis to bool for masking
        masked_distance = torch.masked_select(squared_distance, keypoint_vis.squeeze(-1).to(dtype=torch.bool))

        # If there are no valid keypoints, return 0 to avoid division by zero
        if masked_distance.numel() == 0:
            return torch.tensor(0.0, device=squared_distance.device)

        # Calculate the average to get the overall L2 loss
        total_loss = torch.mean(masked_distance)

        return total_loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):
        # Calculate the absolute difference
        absolute_difference = torch.sum(torch.abs(pre_xyz - gt_xyz), dim=2)

        # Convert keypoint_vis to bool for masking
        masked_distance = torch.masked_select(absolute_difference, keypoint_vis.squeeze(-1).to(dtype=torch.bool))

        # If there are no valid keypoints, return 0 to avoid division by zero
        if masked_distance.numel() == 0:
            return torch.tensor(0.0, device=absolute_difference.device)

        # Calculate the average to get the overall L1 loss
        total_loss = torch.mean(masked_distance)

        return total_loss



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class LossCalculation(nn.Module):
    def __init__(self, device = 'cpu', loss_type = 'L2', comp_xyz_loss = True, comp_uv_loss = True, comp_contrastive_loss = False):
        super(LossCalculation, self).__init__()
        assert loss_type in ['L2', 'L2']
        self.device = device
        self.loss_type = loss_type
        self.comp_xyz_loss = comp_xyz_loss
        self.comp_uv_loss = comp_uv_loss
        self.comp_contrastive_loss = comp_contrastive_loss

        if loss_type == 'L2':
            self.LossObj = L2Loss()
        else:
            self.LossObj = L1Loss()
        if comp_contrastive_loss:
            self.ContrastiveLossObj = ContrastiveLoss()

    def compute_3d_coord_loss(self, pre_xyz, gt_xyz, keypoint_vis):
        return self.LossObj(pre_xyz, gt_xyz, keypoint_vis)

    def compute_uv_coord_loss(self, pre_uv, gt_uv, keypoint_vis):
        return self.LossObj(pre_uv, gt_uv, keypoint_vis)
    
    def compute_contrastive_loss(self, feat1, feat2, label):
        return self.ContrastiveLossObj(feat1, feat2, label)
    
    def forward(self, pre_xyz, gt_xyz, pre_uv, gt_uv, keypoint_vis, feat1 = None, feat2 = None, label = None):
        if self.comp_xyz_loss:
            loss_xyz = self.compute_3d_coord_loss(pre_xyz, gt_xyz, keypoint_vis)
        else:
            loss_xyz = torch.tensor(0)

        if self.comp_uv_loss:
            loss_uv = self.compute_uv_coord_loss(pre_uv, gt_uv, keypoint_vis)
        else:
            loss_uv = torch.tensor(0)

        if self.comp_contrastive_loss:
            loss_contrast = self.compute_contrastive_loss(feat1, feat2, label)
        else:
            loss_contrast = torch.tensor(0)
        
        # loss = loss_xyz + loss_uv + loss_contrast
        return loss_xyz, loss_uv, loss_contrast


if __name__ == '__main__':
    # Use L2Loss class for testing
    pre_xyz = torch.ones(10, 21, 3) # pre_xyz with all elements being 1
    gt_xyz = torch.zeros(10, 21, 3) # gt_xyz with all elements 0
    keypoint_vis = torch.ones(10, 21, 1) # Assume all keypoints are visible
    keypoint_vis = torch.zeros(10, 21, 1) # Assume all keypoints are invisible

    # Initialize L2Loss instance
    l2_loss = L2Loss()

    # Calculate loss
    loss = l2_loss(pre_xyz, gt_xyz, keypoint_vis)
    print(loss)
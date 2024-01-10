import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):
        # Calculate the square of the Euclidean distance
        squared_distance = torch.sum((pre_xyz - gt_xyz) ** 2, dim=2) # Calculate along the third dimension

        # Select valid key points through mask
        masked_distance = squared_distance * keypoint_vis.squeeze(-1) # Remove the extra dimension of the mask

        # Calculate L2 loss for each valid keypoint
        loss_per_keypoint = torch.mean(masked_distance, dim=1) # Calculate along the second dimension

        # Calculate the average to get the overall L2 loss
        total_loss = torch.mean(loss_per_keypoint)

        return total_loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pre_xyz, gt_xyz, keypoint_vis):
        # Calculate the absolute difference
        absolute_difference = torch.abs(pre_xyz - gt_xyz)

        # Select valid key points through mask
        masked_difference = absolute_difference * keypoint_vis.squeeze(-1) # Remove the extra dimension of the mask

        # Calculate L1 loss for each valid keypoint
        loss_per_keypoint = torch.mean(masked_difference, dim=1) # Calculate along the second dimension

        # Calculate the average to get the overall L1 loss
        total_loss = torch.mean(loss_per_keypoint)

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
    def __init__(self, device = 'cpu', loss_type = 'L2', uv_loss = True, contrastive_loss = False):
        super(LossCalculation, self).__init__()
        assert loss_type in ['L2', 'L2']
        self.device = device
        self.loss_type = loss_type
        self.uv_loss = uv_loss
        self.contrastive_loss = contrastive_loss

        if loss_type == 'L2':
            self.LossObj = L2Loss()
        else:
            self.LossObj = L1Loss()
        if contrastive_loss:
            self.ContrastiveLossObj = ContrastiveLoss()

    def compute_3d_coord_loss(self, pre_xyz, gt_xyz, keypoint_vis):
        return self.LossObj(pre_xyz, gt_xyz, keypoint_vis)

    def compute_uv_coord_loss(self, pre_uv, gt_uv, keypoint_vis):
        return self.LossObj(pre_uv, gt_uv, keypoint_vis)
    
    def compute_contrastive_loss(self, feat1, feat2, label):
        return self.ContrastiveLossObj(feat1, feat2, label)
    
    def forward(self, pre_xyz, gt_xyz, pre_uv, gt_uv, keypoint_vis, feat1 = None, feat2 = None, label = None):
        loss_xyz = self.compute_3d_coord_loss(self, pre_xyz, gt_xyz, keypoint_vis)

        if self.uv_loss:
            loss_uv = self.compute_uv_coord_loss(self, pre_uv, gt_uv, keypoint_vis)
        else:
            loss_uv = 0

        if self.contrastive_loss:
            loss_contrast = self.compute_contrastive_loss(feat1, feat2, label)
        else:
            loss_contrast = 0
        
        loss = loss_xyz + loss_uv + loss_contrast
        return loss
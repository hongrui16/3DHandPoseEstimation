import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config

from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.sub_modules.bonePrediction import BoneAnglePrediction, BoneLengthPrediction



class TwoDimHandPose(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(TwoDimHandPose, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(config.resnet_out_feature_dim)
        self.twoDimPoseEstimate = torch.nn.Sequential(
            torch.nn.Linear(config.resnet_out_feature_dim, config.resnet_out_feature_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//2, config.resnet_out_feature_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//4, config.resnet_out_feature_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//8, config.resnet_out_feature_dim//16),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//16, config.eypoint_num*2),#[x1, y1, x2, y2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )
        self.diffusion_loss = torch.tensor(0, device=device)

    def forward(self, img, camera_intrinsic_matrix = None, index_root_bone_length = None, keypoint_xyz_root = None, pose_x0 = None):
        # Extract features using ResNet model
        resnet_features = self.resnet_extractor(img)
        pose = self.twoDimPoseEstimate(resnet_features)
        # Get the batch size and the number of features
        b, n = pose.shape
        b, c, h, w = img.shape

        # Reshape the features to [bs, n, 2]
        pose = pose.view(b, -1, 2)
        '''
        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            ...
        ]
        '''
        # Calculate u and v based on w and h
        u = pose[:, :, 0] * w
        v = pose[:, :, 1] * h

        # Concatenate u and v along the last dimension
        keypoint_uv21 = torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1)


        refined_joint_coord = [None, keypoint_uv21, None]
        return refined_joint_coord, self.diffusion_loss, [None, None]
        
        




import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config

from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.sub_modules.bonePrediction import BoneAnglePrediction, BoneLengthPrediction



class TwoDimHandPoseEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(TwoDimHandPoseEstimation, self).__init__()
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
        return refined_joint_coord, torch.tensor(0)
        
        




class TwoDimHandPoseWithFKEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(TwoDimHandPoseWithFKEstimation, self).__init__()
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
            torch.nn.Linear(config.resnet_out_feature_dim//16, config.keypoint_num*2),#[x1, y1, x2, y2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )

        self.boneAngle = BoneAnglePrediction(input_dim = config.keypoint_num*2)
        self.bonelength = BoneLengthPrediction(input_dim = config.keypoint_num*2)
        self.forwardKenematic = ForwardKinematics(device)



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

        keypoints = keypoint_uv21.view(b, -1)

        root_angles, other_angles = self.boneAngle(keypoints)
        bone_lengths = self.bonelength(keypoints)

        keypoint_xyz21_absolute, kp_coord_uv21_from_pro, _ = self.forwardKenematic(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root)
        
        if config.is_inference:
            return [keypoint_xyz21_absolute, kp_coord_uv21_from_pro, keypoint_uv21], torch.tensor(0)
        
        if config.uv_from_xD == 2.5:
            keypoint_uv21 = (keypoint_uv21 + kp_coord_uv21_from_pro )/2
        elif config.uv_from_xD == 3:
            keypoint_uv21 = kp_coord_uv21_from_pro
        elif config.uv_from_xD == 2:
            keypoint_uv21 = keypoint_uv21

        return [keypoint_xyz21_absolute, keypoint_uv21, None], torch.tensor(0)
        
        



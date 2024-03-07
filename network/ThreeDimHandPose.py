import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config

from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.sub_modules.bonePrediction import BoneAnglePrediction, BoneLengthPrediction
from utils.coordinate_trans import batch_project_xyz_to_uv




class ThreeDimHandPose(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(ThreeDimHandPose, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(config.resnet_out_feature_dim)
        self.threeDimPoseEstimate = torch.nn.Sequential(
            torch.nn.Linear(config.resnet_out_feature_dim, config.resnet_out_feature_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//2, config.resnet_out_feature_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//4, config.resnet_out_feature_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(config.resnet_out_feature_dim//8, config.keypoint_num*3),#[x1, y1, z1, x2, y2, z2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )
        self.forward_kinematics_module = ForwardKinematics(device = device)
        self.bone_angle_pred_model = BoneAnglePrediction()
        self.bone_length_pred_model = BoneLengthPrediction()
        self.diffusion_loss = torch.tensor(0, device=device)
    

    def forward(self, img, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root, pose_x0 = None):
        resnet_features = self.resnet_extractor(img)
        pose3D_xyz = (self.threeDimPoseEstimate(resnet_features) - 0.5 )*2
        root_angles, other_angles = self.bone_angle_pred_model(pose3D_xyz)
        bone_lengths = self.bone_length_pred_model(pose3D_xyz)
        refined_joint_coord = self.forward_kinematics_module(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

        # refined_joint_coord ## [positions_xyz, positions_uv]
        return refined_joint_coord, self.diffusion_loss, [None, None]


if __name__ == "__main__": 
    pass
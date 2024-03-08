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
from utils.util import build_sequtial

class OnlyThreeDimHandPose(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(OnlyThreeDimHandPose, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(config.resnet_out_feature_dim)
        sequential = build_sequtial(config.resnet_out_feature_dim, config.keypoint_num*3, 2, activation='LeakyReLU', use_sigmoid=False)
        # self.threeDimPoseEstimate = torch.nn.Sequential(
        #     torch.nn.Linear(config.resnet_out_feature_dim, config.resnet_out_feature_dim//2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(config.resnet_out_feature_dim//2, config.resnet_out_feature_dim//4),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(config.resnet_out_feature_dim//4, config.resnet_out_feature_dim//8),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(config.resnet_out_feature_dim//8, config.keypoint_num*3),#[x1, y1, z1, x2, y2, z2......] the ration of u v, x=u/width, y=v/height
        #     torch.nn.Sigmoid()
        # )
        self.threeDimPoseEstimate = torch.nn.Sequential(*sequential)

    def forward(self, img, camera_intrinsic_matrix, index_root_bone_length = None, kp_coord_xyz_root = None, pose_x0 = None):
        resnet_features = self.resnet_extractor(img)
        pose3D_xyz = (self.threeDimPoseEstimate(resnet_features) - 0.5 )*2
        b, n = pose3D_xyz.shape
        pose3D_xyz = pose3D_xyz.view(b, -1, 3)
        uv21 = batch_project_xyz_to_uv(pose3D_xyz, camera_intrinsic_matrix)

        refined_joint_coord = [pose3D_xyz, uv21, None]
        return refined_joint_coord, torch.tensor(0)
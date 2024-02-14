import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config.config import *

from network.sub_modules.conditionalDiffusion import *
from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.sub_modules.bonePrediction import BoneAnglePrediction, BoneLengthPrediction
from utils.coordinate_trans import batch_project_xyz_to_uv




class ThreeDimHandPoseEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(ThreeDimHandPoseEstimation, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(resnet_out_feature_dim)
        self.threeDimPoseEstimate = torch.nn.Sequential(
            torch.nn.Linear(resnet_out_feature_dim, resnet_out_feature_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//2, resnet_out_feature_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//4, resnet_out_feature_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//8, keypoint_num*3),#[x1, y1, z1, x2, y2, z2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )
        self.forward_kinematics_module = ForwardKinematics(device = device)
        self.bone_angle_pred_model = BoneAnglePrediction()
        self.bone_length_pred_model = BoneLengthPrediction()
    

    def forward(self, img, camera_intrinsic_matrix, pose_x0, index_root_bone_length, kp_coord_xyz_root):
        resnet_features = self.resnet_extractor(img)
        pose3D_xyz = (self.threeDimPoseEstimate(resnet_features) - 0.5 )*2
        root_angles, other_angles = self.bone_angle_pred_model(pose3D_xyz)
        bone_lengths = self.bone_length_pred_model(pose3D_xyz)
        refined_joint_coord = self.forward_kinematics_module(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

        # refined_joint_coord ## [positions_xyz, positions_uv]
        return refined_joint_coord, torch.tensor(0)



class OnlyThreeDimHandPoseEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(OnlyThreeDimHandPoseEstimation, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(resnet_out_feature_dim)
        self.threeDimPoseEstimate = torch.nn.Sequential(
            torch.nn.Linear(resnet_out_feature_dim, resnet_out_feature_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//2, resnet_out_feature_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//4, resnet_out_feature_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//8, keypoint_num*3),#[x1, y1, z1, x2, y2, z2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )
    

    def forward(self, img, camera_intrinsic_matrix, pose_x0 = None, index_root_bone_length = None, kp_coord_xyz_root = None):
        resnet_features = self.resnet_extractor(img)
        pose3D_xyz = (self.threeDimPoseEstimate(resnet_features) - 0.5 )*2
        b, n = pose3D_xyz.shape
        pose3D_xyz = pose3D_xyz.view(b, -1, 3)
        uv21 = batch_project_xyz_to_uv(pose3D_xyz, camera_intrinsic_matrix)

        refined_joint_coord = [pose3D_xyz, uv21]
        return refined_joint_coord, torch.tensor(0)
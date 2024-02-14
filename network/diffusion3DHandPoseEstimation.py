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




class Diffusion3DHandPoseEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(Diffusion3DHandPoseEstimation, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(condition_feat_dim)
        self.diff_model = DiffusionJointEstimation()
        self.forward_kinematics_module = ForwardKinematics(device = device)
        self.bone_angle_pred_model = BoneAnglePrediction()
        self.bone_length_pred_model = BoneLengthPrediction()
    
    def compute_diffusion_loss(self, pose_x0, resnet_features):
        '''
            pose_x0: groundtruth xyz coordinates
        '''
        # Concatenate image features, 3D points, and time-step information
        # You may need to adjust the dimensions for concatenation based on your specific inputs.
        return self.diff_model.diffusion_loss(pose_x0, resnet_features)

    def joint_coord_by_diffusion(self, resnet_features):
        coarse_joint_coord = self.diff_model.joint_coord_sampling(resnet_features)
        coarse_joint_coord = coarse_joint_coord.squeeze()
        # joint_pose = joint_pose.reshape(batch_size, -1, 3)
        return coarse_joint_coord
    
    def forward(self, img, camera_intrinsic_matrix, pose_x0, index_root_bone_length, kp_coord_xyz_root):
        resnet_features = self.resnet_extractor(img)
        coarse_joint_coord = self.joint_coord_by_diffusion(resnet_features)
        root_angles, other_angles = self.bone_angle_pred_model(coarse_joint_coord)
        bone_lengths = self.bone_length_pred_model(coarse_joint_coord)
        refined_joint_coord = self.forward_kinematics_module(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

        diffusion_loss = self.compute_diffusion_loss(pose_x0, resnet_features)
        # refined_joint_coord ## [positions_xyz, positions_uv]
        return refined_joint_coord, diffusion_loss



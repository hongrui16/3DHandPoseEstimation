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



class TwoDimHandPoseEstimation(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(TwoDimHandPoseEstimation, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(resnet_out_feature_dim)
        self.twoDimPoseEstimate = torch.nn.Sequential(
            torch.nn.Linear(resnet_out_feature_dim, resnet_out_feature_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//2, resnet_out_feature_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//4, resnet_out_feature_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//8, resnet_out_feature_dim//16),
            torch.nn.ReLU(),
            torch.nn.Linear(resnet_out_feature_dim//16, keypoint_num*2),#[x1, y1, x2, y2......] the ration of u v, x=u/width, y=v/height
            torch.nn.Sigmoid()
        )
    


    def forward(self, img, camera_intrinsic_matrix = None, pose_x0 = None, index_root_bone_length = None, keypoint_xyz_root = None):
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

        refined_joint_coord = [None, keypoint_uv21]
        return refined_joint_coord, torch.tensor(0)
        
        



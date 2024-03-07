import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import pickle
import numpy as np
import sys
import os

import torch.nn.functional as F
import torchvision.models as models

sys.path.append("../..")
from config import config
from utils.util import *
from network.sub_modules.MANOLayer import ManoLayer



class ExtendedResNet50(nn.Module):
    def __init__(self):
        super(ExtendedResNet50, self).__init__()
        # Load a pre-trained ResNet18 model
        self.feature_extractor = models.resnet50(pretrained=True)
        # Modify conv1 for 21 input channels and change kernel size to 3
        self.feature_extractor.conv1 = nn.Conv2d(24, self.feature_extractor.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Utilize the original fully connected layer, get the output channels
        self.num_output_features = self.feature_extractor.fc.out_features
    
    def forward(self, x):
        # print('1 x.shape', x.shape)
        x = self.feature_extractor(x)  # Extract features through the modified ResNet18
        # print('2 x.shape', x.shape)
        return x


class Resnet50MANO(nn.Module):
    def __init__(self, device = 'cpu', mano_right_hand_path = None):
        super(Resnet50MANO, self).__init__()
        self.extended_resnet50_extractor = ExtendedResNet50()
        num_output_features = self.extended_resnet50_extractor.num_output_features

        if config.network_regress_uv:
            fc_dim = 10 + config.mano_pose_num + 3 + 3 # 10 for belta, config.mano_pose_num for theta, 3 for rot, and 3 for scale and translation
            self.mean = Variable(torch.FloatTensor([545.,128.,128.,]).to(device))
        else:
            fc_dim = 10 + config.mano_pose_num + 3 # 10 for belta, config.mano_pose_num for theta, 3 for rot

        sequential = build_sequtial(num_output_features, fc_dim, 2, activation='ReLU', use_sigmoid=True)
        # sequential = build_sequtial(num_output_features, fc_dim, 2, activation='LeakyReLU', use_sigmoid=False)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)
        self.mano_layer = ManoLayer(device, mano_right_hand_path, pose_num=config.mano_pose_num)


    def forward(self, x):
        """Forward pass through PosePrior.

        Args:
            x - (batch x 24 x 256 x 256): 2D keypoint heatmaps.

        Returns:
            uv21_2d: [batch, num_keypoints * 3]: xyz coordinates of the hand in canonical 3D space.
        """
        x = self.extended_resnet50_extractor(x)
        xs = self.mlp(x)
        
        rot = xs[:,0:3]    
        theta = xs[:,3:config.mano_pose_num+3]
        beta = xs[:,config.mano_pose_num+3:config.mano_pose_num+13] 

        rot = (rot - 1/2) * 2 * math.pi
        theta = (theta - 1/2) * 4
        beta = (beta - 1/2) * 0.1
        vertices_3d, joint21_3d = self.mano_layer(rot,theta,beta)
        if config.network_regress_uv:
            scale = xs[:,-3] + self.mean[:,0:1]
            trans = xs[:,-2:] + self.mean[:,1:]
            uv21_2d = trans.unsqueeze(1) + scale.unsqueeze(1).unsqueeze(2) * joint21_3d[:,:,:2] 
            uv21_2d = uv21_2d.view(uv21_2d.size(0),-1)      
        else:
            uv21_2d = None        
        #x3d = scale.unsqueeze(1).unsqueeze(2) * x3d
        #x3d[:,:,:2]  = trans.unsqueeze(1) + x3d[:,:,:2] 
        
        return joint21_3d, uv21_2d, theta, beta


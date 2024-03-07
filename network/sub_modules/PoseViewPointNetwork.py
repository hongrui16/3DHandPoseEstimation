import math

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

sys.path.append("../..")
from config import config
from utils.util import *



class ExtendedResNet18(nn.Module):
    def __init__(self):
        super(ExtendedResNet18, self).__init__()
        # Load a pre-trained ResNet18 model
        self.feature_extractor = models.resnet18(pretrained=True)
        # Modify conv1 for 21 input channels and change kernel size to 3
        self.feature_extractor.conv1 = nn.Conv2d(config.input_channels, self.feature_extractor.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Utilize the original fully connected layer, get the output channels
        self.num_output_features = self.feature_extractor.fc.out_features
    
    def forward(self, x):
        # print('1 x.shape', x.shape)
        x = self.feature_extractor(x)  # Extract features through the modified ResNet18
        # print('2 x.shape', x.shape)
        return x

class PosePrior(nn.Module):
    def __init__(self):
        super(PosePrior, self).__init__()
        self.extended_resnet18_extractor = ExtendedResNet18()
        num_output_features = self.extended_resnet18_extractor.num_output_features

        output_dim = 63 # 21 * 3
        sequential = build_sequtial(num_output_features, output_dim, 2, activation='LeakyReLU', use_sigmoid=False)
        # sequential = build_sequtial(num_output_features, output_dim, 2, activation='LeakyReLU', use_sigmoid=True)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)

    def forward(self, x):
        """Forward pass through PosePrior.

        Args:
            x - (batch x 21 x 256 x 256): 2D keypoint heatmaps.

        Returns:
            (batch x num_keypoints x 3): xyz coordinates of the hand in
                canonical 3D space.
        """
        x = self.extended_resnet18_extractor(x)
        x = self.mlp(x)
        
        return x



class ViewPoint(nn.Module):
    def __init__(self):
        super(ViewPoint, self).__init__()
        self.extended_resnet18_extractor = ExtendedResNet18()
        num_output_features = self.extended_resnet18_extractor.num_output_features

        output_dim = 3 # 
        sequential = build_sequtial(num_output_features, output_dim, 4, activation='LeakyReLU', use_sigmoid=False)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)

    def forward(self, x):
        """Forward pass through PosePrior.

        Args:
            x - (batch x 21 x 256 x 256): 2D keypoint heatmaps.

        Returns:
            (batch x 1): ux, uy, uz- angles of the hand in x, y, z axis.
        """

        x = self.extended_resnet18_extractor(x)
        x = self.mlp(x)
        # x = (x - 0.5)* 2 * math.pi
        ux, uy, uz = x[:, 0], x[:, 1], x[:, 2]
        ux = ux.unsqueeze(1)
        uy = uy.unsqueeze(1)
        uz = uz.unsqueeze(1)
        return ux, uy, uz


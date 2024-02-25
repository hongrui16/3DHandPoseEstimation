import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../..")
from config import config

from utils.util import *




class Pose3dPrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = None):
        super(Pose3dPrediction, self).__init__()        
        sequential = build_sequtial(input_dim, config.keypoint_num*3, 4)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)

    def forward(self, x):
        kps = self.mlp(x)
        kps = (kps - 0.5)*4
        return kps


class ViewPointPrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = None):
        super(ViewPointPrediction, self).__init__()
        rot_dim = 64
        sequential = build_sequtial(input_dim, rot_dim, 4)
        #Create Sequential model
        self.mlp = torch.nn.Sequential(*sequential)
        self.fc_vp_ux = nn.Linear(rot_dim, 1)
        self.fc_vp_uy = nn.Linear(rot_dim, 1)
        self.fc_vp_uz = nn.Linear(rot_dim, 1)
        
    def forward(self, x):
        angles = self.mlp(x)
        # Scale root_angles to the range of [-π, π]

        angles = (angles - 0.5)* 2 * math.pi

        ux = self.fc_vp_ux(angles)
        uy = self.fc_vp_uy(angles)
        uz = self.fc_vp_uz(angles)


        return ux, uy, uz
    





if __name__ == "__main__":


    pass

    
    
    # print('betas:', betas)
    # print('pose:', pose)
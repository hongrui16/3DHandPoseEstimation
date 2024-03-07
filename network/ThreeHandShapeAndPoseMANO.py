import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from torch import nn

sys.path.append("..")

from config import config
from network.sub_modules.resnetMANO import ResNet_Mano, BasicBlock


class ThreeHandShapeAndPoseMANO(torch.nn.Module):
    def __init__(self, device = 'cpu', input_option=1, mano_right_hand_path = None):
        super(ThreeHandShapeAndPoseMANO, self).__init__()
        self.device = device
        if mano_right_hand_path is None:
            mano_right_hand_path = config.mano_right_hand_path
        self.resnet_Mano = ResNet_Mano(device, BasicBlock, [3, 4, 6, 3], mano_right_hand_path = mano_right_hand_path)    
        self.diffusion_loss = torch.tensor(0, device=device)

    def forward(self, img, camera_intrinsic_matrix = None, index_root_bone_length = None, kp_coord_xyz_root = None, pose_x0 = None):
        joint21_3d, uv21_2d = self.resnet_Mano(img)
        refined_joint_coord = [joint21_3d, uv21_2d, None]
        # refined_joint_coord ## [positions_xyz, positions_uv]
        return refined_joint_coord, self.diffusion_loss, [None, None]



if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1
    MANO_RIGHT_pkl = '../config/mano/models/MANO_RIGHT.pkl'

    model = ThreeHandShapeAndPoseMANO(device, mano_right_hand_path=MANO_RIGHT_pkl).to(device)
    image = torch.rand(batch_size, 3, 320, 320, device = device)*255
    camera_intrinsic_matrix = torch.rand(batch_size, 3, 3, device = device)*400
    index_root_bone_length = torch.rand(batch_size, 1, device = device)
    kp_coord_xyz_root = torch.rand(batch_size, 3, device = device)

    refined_joint_coord, _ = model(image, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

    joint_xyz21, uv21, _ = refined_joint_coord
    print('joint_xyz21:', joint_xyz21)
    print('uv21:', uv21)


    
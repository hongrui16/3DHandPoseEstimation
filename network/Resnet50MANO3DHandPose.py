

'''
this file is based on the paper '3D Hand Shape and Pose from Images in the Wild'(https://arxiv.org/abs/1902.03451)
The code is different implemented.
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config


from network.sub_modules.resnet50MANO import Resnet50MANO
from utils.coordinate_trans import batch_project_xyz_to_uv




class Resnet50MANO3DHandPose(torch.nn.Module):
    def __init__(self, device = 'cpu', mano_right_hand_path = None):
        super(Resnet50MANO3DHandPose, self).__init__()
        self.device = device
        if mano_right_hand_path is None:
            mano_right_hand_path = config.mano_right_hand_path
        self.mano_model = Resnet50MANO(device=device, mano_right_hand_path = mano_right_hand_path)
        self.diffusion_loss = torch.tensor(0, device=device)
        
    

    def match_mano_to_RHD(self, mano_joints, index_root_bone_length,  kp_coord_xyz_root):
        # mano_joints: [batch, 21, 3]
        # return: [batch, 21, 3]
        if not config.joint_order_switched:
            # print('config.joint_order_switched', config.joint_order_switched)
            for i in range(1, 21, 4):
                mano_joints[:,[i, i + 3]] = mano_joints[:,[i + 3, i]]
                mano_joints[:,[i + 1, i + 2]] = mano_joints[:,[i + 2, i + 1]]
        # print('mano_joints:', mano_joints.shape)
        mano_joints_root = mano_joints[:, 0, :]  # this is the palm coord
        mano_joints_root = mano_joints_root.unsqueeze(1)  # [bs, 3] -> [bs, 1, 3]
        # print('mano_joints_root:', mano_joints_root.shape)
        mano_joints_rel = mano_joints - mano_joints_root # relative coords in metric coords
        # print('mano_joints_rel:', mano_joints_rel)
        scale = torch.sqrt((mano_joints_rel[:, 12, :]).pow(2).sum(dim=-1))
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [batch] -> [batch, 1, 1]
        # print('scale:', scale)
        # print('mano_joints_rel:', mano_joints_rel)
        mano_joints_rel_normalized = mano_joints_rel / scale ##normalized by length of 12->0
        # print('mano_joints_rel_normalized:', mano_joints_rel_normalized)
        
        kp_coord_xyz_root = kp_coord_xyz_root.unsqueeze(1)  # [bs, 3] -> [bs, 1, 3]
        index_root_bone_length = index_root_bone_length.unsqueeze(-1)  # [bs, 1] -> [bs, 1, 1]
        joint_xyz21 = mano_joints_rel_normalized * index_root_bone_length + kp_coord_xyz_root

        return mano_joints_rel_normalized, joint_xyz21

    def forward(self, img, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root, pose_x0 = None):
        # img: [batch, 24, 320, 320]
        # camera_intrinsic_matrix: [batch, 3, 3]
        # index_root_bone_length: [batch, 1]
        # kp_coord_xyz_root: [batch, 3]
        # pose_x0: [batch, 21, 3]
        joint21_3d, uv21_2d, theta, beta = self.mano_model(img)

        # mano_joints_rel_normalized, joint21_3d = self.match_mano_to_RHD(joint21_3d, index_root_bone_length,  kp_coord_xyz_root)
        
        uv21 = batch_project_xyz_to_uv(joint21_3d, camera_intrinsic_matrix)
        refined_joint_coord = [joint21_3d, uv21, None]
        return refined_joint_coord, self.diffusion_loss, [theta, beta]





if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MANO_RIGHT_pkl = '../config/mano/models/MANO_RIGHT.pkl'
    batch_size = 1
    MANO3DHandPose = Resnet50MANO3DHandPose(device, MANO_RIGHT_pkl).to(device)

    batch_size = 1
    image_shape = (batch_size, 3, 256, 256)
    kp_vis21_shape = (batch_size, 21, 1)
    kp_coord_xyz21_shape = (batch_size, 21, 3)
    kp_coord_21_shape = (batch_size, 21, 2)
    scoremap_shape = (batch_size, 21, 256, 256)
    hand_shape = (batch_size, 256, 256)
    camera_intrinsic_matrix_shape = (batch_size, 3, 3)
    kp_xyz_root_shape = (batch_size, 3)
    kp_scale_shape = (batch_size, 1)

    image = torch.zeros(image_shape).to(device) + 0.5
    bs, c, h, w = image.shape
    image[:, :, -h//2:] = -0.5
    keypoint_vis21_gt = torch.ones(kp_vis21_shape, dtype=torch.bool, device=device)
    index_root_bone_length = torch.ones(kp_scale_shape, device=device)
    keypoint_xyz_root = torch.zeros(kp_xyz_root_shape).to(device)
    keypoint_xyz21_gt = torch.zeros(kp_coord_xyz21_shape).to(device) + 0.5
    keypoint_xyz21_gt[:, 0] = 0
    keypoint_xyz21_gt[:, -10:] = -0.5
    keypoint_xyz21_rel_normed_gt = keypoint_xyz21_gt
    scoremap = torch.zeros(scoremap_shape).to(device)
    camera_intrinsic_matrix = torch.zeros(camera_intrinsic_matrix_shape).to(device)
    camera_intrinsic_matrix[:, 0, 0] = 600
    camera_intrinsic_matrix[:, 1, 1] = 600
    camera_intrinsic_matrix[:, 0, 2] = 300
    camera_intrinsic_matrix[:, 1, 2] = 300
    camera_intrinsic_matrix[:, 2, 2] = 1

    if config.input_channels == 24:
        input = torch.cat([image, scoremap], dim=1)
    elif config.input_channels == 21:
        input = scoremap
    elif config.input_channels == 3:
        input = image
    else:
        raise ValueError('input_channels are not supported')

    refined_joint_coord, _, _ = MANO3DHandPose(input, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root)

    joint_xyz21, uv21, _ = refined_joint_coord
    print('joint_xyz21:', joint_xyz21)
    print('uv21:', uv21)


    
    
    # print('betas:', betas)
    # print('pose:', pose)
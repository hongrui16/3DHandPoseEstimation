import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config

from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor

from network.sub_modules.MANOLayer import ManoLayer, MANOBetasPrediction, MANOThetaPrediction
from utils.coordinate_trans import batch_project_xyz_to_uv




class MANO3DHandPose(torch.nn.Module):
    def __init__(self, device = 'cpu', mano_right_hand_path = None):
        super(MANO3DHandPose, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(config.resnet_out_feature_dim)
        self.betas_predictor = MANOBetasPrediction(device, config.resnet_out_feature_dim)
        self.theta_predictor = MANOThetaPrediction(device, config.resnet_out_feature_dim)
        if mano_right_hand_path is None:
            mano_right_hand_path = config.mano_right_hand_path
        self.mano_layer = ManoLayer(device, mano_right_hand_path, pose_num=config.mano_pose_num)
        self.diffusion_loss = torch.tensor(0, device=device)

    def match_mano_to_RHD(self, mano_joints):
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
        return mano_joints_rel_normalized

    def forward(self, img, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root, pose_x0 = None):
        # img: [batch, 3, 320, 320]
        # camera_intrinsic_matrix: [batch, 3, 3]
        # index_root_bone_length: [batch, 1]
        # kp_coord_xyz_root: [batch, 3]
        # pose_x0: [batch, 21, 3]
        resnet_features = self.resnet_extractor(img)
        root_angles, other_angles = self.theta_predictor(resnet_features)
        betas = self.betas_predictor(resnet_features)
        vertices, joint = self.mano_layer(root_angles, other_angles, betas)
        joint_rel_normalized = self.match_mano_to_RHD(joint)
        kp_coord_xyz_root = kp_coord_xyz_root.unsqueeze(1)  # [bs, 3] -> [bs, 1, 3]
        index_root_bone_length = index_root_bone_length.unsqueeze(-1)  # [bs, 1] -> [bs, 1, 1]
        joint_xyz21 = joint_rel_normalized * index_root_bone_length + kp_coord_xyz_root
        uv21 = batch_project_xyz_to_uv(joint_xyz21, camera_intrinsic_matrix)
        refined_joint_coord = [joint_xyz21, uv21, None]
        return refined_joint_coord, self.diffusion_loss, [None, None]





if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MANO_RIGHT_pkl = '../config/mano/models/MANO_RIGHT.pkl'
    batch_size = 1
    MANO3DHandPose = MANO3DHandPose(device, MANO_RIGHT_pkl).to(device)
    image = torch.rand(batch_size, 3, 320, 320, device = device)*255
    camera_intrinsic_matrix = torch.rand(batch_size, 3, 3, device = device)*400
    index_root_bone_length = torch.rand(batch_size, 1, device = device)
    kp_coord_xyz_root = torch.rand(batch_size, 3, device = device)

    refined_joint_coord, _ = MANO3DHandPose(image, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

    joint_xyz21, uv21, _ = refined_joint_coord
    print('joint_xyz21:', joint_xyz21)
    print('uv21:', uv21)


    
    
    # print('betas:', betas)
    # print('pose:', pose)
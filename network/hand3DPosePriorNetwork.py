import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config



from network.sub_modules.PoseViewPointNetwork import PosePrior, ViewPoint
from utils.coordinate_trans import batch_project_xyz_to_uv
from utils.general import _get_rot_mat





class Hand3DPosePriorNetwork(torch.nn.Module):
    def __init__(self):
        super(Hand3DPosePriorNetwork, self).__init__()
        self.PosePrior_net = PosePrior()
        self.ViewPoint_net = ViewPoint()
    
    def forward(self, img, camera_intrinsic_matrix = None, index_root_bone_length = None, kp_coord_xyz_root = None, pose_x0 = None):
        # img: [batch, 3, 320, 320]
        # camera_intrinsic_matrix: [batch, 3, 3]
        # index_root_bone_length: [batch, 1]
        # kp_coord_xyz_root: [batch, 3]
        # pose_x0: [batch, 21, 3]
        can_xyz_kps21 = self.PosePrior_net(img)
        b, n = can_xyz_kps21.shape
        can_xyz_kps21 = can_xyz_kps21.view(b, -1, 3)
        ux, uy, uz = self.ViewPoint_net(img)
        # assemble rotation matrix
        rot_mat = _get_rot_mat(ux, uy, uz)
        # Assuming can_xyz_kps21 and rot_mat are PyTorch tensors with shapes [B, 21, 3] and [B, 3, 3], respectively.
        coord_xyz_rel_normed = torch.matmul(can_xyz_kps21, rot_mat) #[B, 21, 3]
        # print('config.is_inference', config.is_inference)
        if config.is_inference:
            kp_coord_xyz_root = kp_coord_xyz_root.unsqueeze(1)  # [bs, 3] -> [bs, 1, 3]
            index_root_bone_length = index_root_bone_length.unsqueeze(-1)  # [bs, 1] -> [bs, 1, 1]
            joint_xyz21 = coord_xyz_rel_normed * index_root_bone_length + kp_coord_xyz_root
            uv21 = batch_project_xyz_to_uv(joint_xyz21, camera_intrinsic_matrix)
            result = [joint_xyz21, uv21, None], None
        else:
            result = [coord_xyz_rel_normed, can_xyz_kps21, rot_mat], None
        return result






if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Hand3DPosePriorNetwork().to(device)

    batch_size = 1
    image = torch.rand(batch_size, 21, 320, 320, device = device)*255
    camera_intrinsic_matrix = torch.rand(batch_size, 3, 3, device = device)*400
    index_root_bone_length = torch.rand(batch_size, 1, device = device)
    kp_coord_xyz_root = torch.rand(batch_size, 3, device = device)

    refined_joint_coord, _ = model(image, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)

    joint_xyz21, uv21, _ = refined_joint_coord
    print('joint_xyz21:', joint_xyz21)
    print('uv21:', uv21)

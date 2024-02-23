import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("..")

from config import config

from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor


from network.sub_modules.PoseViewPointEst import *
from utils.coordinate_trans import batch_project_xyz_to_uv




class Hand3DPoseNet(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(Hand3DPoseNet, self).__init__()
        self.device = device
        self.resnet_extractor = ResNetFeatureExtractor(config.resnet_out_feature_dim)
        self.pose_predictor = Pose3dPrediction(device, config.resnet_out_feature_dim)
        self.view_point_predictor = ViewPointPrediction(device, config.resnet_out_feature_dim)
    


    def forward(self, img, camera_intrinsic_matrix = None, index_root_bone_length = None, kp_coord_xyz_root = None, pose_x0 = None):
        # img: [batch, 3, 320, 320]
        # camera_intrinsic_matrix: [batch, 3, 3]
        # index_root_bone_length: [batch, 1]
        # kp_coord_xyz_root: [batch, 3]
        # pose_x0: [batch, 21, 3]
        resnet_features = self.resnet_extractor(img)
        can_xyz_kps21 = self.pose_predictor(resnet_features)
        b, n = can_xyz_kps21.shape
        can_xyz_kps21 = can_xyz_kps21.view(b, -1, 3)
        ux, uy, uz = self.view_point_predictor(resnet_features)
        # assemble rotation matrix
        rot_mat = self._get_rot_mat(ux, uy, uz)
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
            result = [coord_xyz_rel_normed, can_xyz_kps21, rot_mat]
        return result



    def _get_rot_mat(self, ux_b, uy_b, uz_b):
        """Returns a rotation matrix from axis and (encoded) angle in PyTorch."""
        u_norm = torch.sqrt(ux_b**2 + uy_b**2 + uz_b**2 + 1e-8)
        theta = u_norm

        # some tmp vars
        st_b = torch.sin(theta)
        ct_b = torch.cos(theta)
        one_ct_b = 1.0 - torch.cos(theta)

        st = st_b[:, 0]
        ct = ct_b[:, 0]
        one_ct = one_ct_b[:, 0]
        norm_fac = 1.0 / u_norm[:, 0]
        ux = ux_b[:, 0] * norm_fac
        uy = uy_b[:, 0] * norm_fac
        uz = uz_b[:, 0] * norm_fac

        trafo_matrix = self._stitch_mat_from_vecs([ct+ux*ux*one_ct, ux*uy*one_ct-uz*st, ux*uz*one_ct+uy*st,
                                                   uy*ux*one_ct+uz*st, ct+uy*uy*one_ct, uy*uz*one_ct-ux*st,
                                                   uz*ux*one_ct-uy*st, uz*uy*one_ct+ux*st, ct+uz*uz*one_ct])

        return trafo_matrix


    @staticmethod
    def _stitch_mat_from_vecs(vector_list):
        """Stitches a given list of vectors into a 3x3 matrix in PyTorch.

        Input:
            vector_list: list of 9 tensors, which will be stitched into a matrix. The list contains matrix elements
                         in a row-first fashion (m11, m12, m13, m21, m22, m23, m31, m32, m33). The length of the vectors has
                         to be the same, because it is interpreted as batch dimension.
        """
        
        assert len(vector_list) == 9, "There have to be exactly 9 tensors in vector_list."
        batch_size = vector_list[0].shape[0]
        
        # Reshape each tensor in vector_list to have shape [batch_size, 1] and concatenate them along dim=1
        vector_list = [x.view(batch_size, 1) for x in vector_list]
        trafo_matrix = torch.cat(vector_list, dim=1)
        
        # Reshape to have shape [batch_size, 3, 3]
        trafo_matrix = trafo_matrix.view(batch_size, 3, 3)

        return trafo_matrix



if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    


    
    
    # print('betas:', betas)
    # print('pose:', pose)
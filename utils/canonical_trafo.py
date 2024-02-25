#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# import tensorflow as tf
import torch
from torch import nn


def atan2_pytorch(y, x):
    """ My implementation of atan2 in PyTorch. Returns in -pi .. pi."""
    tan = torch.atan(y / (x + 1e-8))  # this returns in -pi/2 .. pi/2
    
    one_map = torch.ones_like(tan)

    # correct quadrant error
    correction = torch.where(x + 1e-8 < 0.0, 3.141592653589793 * one_map, 0.0 * one_map)
    tan_c = tan + correction  # this returns in -pi/2 .. 3pi/2

    # bring to positive values
    correction = torch.where(tan_c < 0.0, 2 * 3.141592653589793 * one_map, 0.0 * one_map)
    tan_zero_2pi = tan_c + correction  # this returns in 0 .. 2pi

    # make symmetric
    correction = torch.where(tan_zero_2pi > 3.141592653589793, -2 * 3.141592653589793 * one_map, 0.0 * one_map)
    tan_final = tan_zero_2pi + correction  # this returns in -pi .. pi
    return tan_final

def _stitch_mat_from_vecs(vector_list):
    """Stitches a given list of vectors into a 3x3 matrix in PyTorch.

    Input:
        vector_list: list of 9 tensors, which will be stitched into a matrix. The list contains matrix elements
            in a row-first fashion (m11, m12, m13, m21, m22, m23, m31, m32, m33). The length of the vectors has
            to be the same, because it is interpreted as batch dimension.
    """
    
    assert len(vector_list) == 9, "There have to be exactly 9 tensors in vector_list."
    
    # Ensure all tensors are the correct shape and stacked along a new dimension
    vector_list = [x.view(-1, 1) for x in vector_list]
    # Concatenate along the new dimension to form a single tensor
    matrix = torch.cat(vector_list, dim=1)
    # Reshape to batch_size x 3 x 3
    batch_size = vector_list[0].size(0)
    trafo_matrix = matrix.view(batch_size, 3, 3)

    return trafo_matrix


def _get_rot_mat_x(angle):
    """Returns a 3D rotation matrix around the x-axis in PyTorch."""
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([one_vec, zero_vec, zero_vec,
                                                  zero_vec, torch.cos(angle), -torch.sin(angle),
                                                  zero_vec, torch.sin(angle), torch.cos(angle)])
    return trafo_matrix


def _get_rot_mat_y(angle):
    """Returns a 3D rotation matrix around the y-axis in PyTorch."""
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), zero_vec, torch.sin(angle),
                                                  zero_vec, one_vec, zero_vec,
                                                  -torch.sin(angle), zero_vec, torch.cos(angle)])
    return trafo_matrix


def _get_rot_mat_z(angle):
    """Returns a 3D rotation matrix around the z-axis in PyTorch."""
    one_vec = torch.ones_like(angle)
    zero_vec = torch.zeros_like(angle)
    trafo_matrix = _stitch_mat_from_vecs([torch.cos(angle), -torch.sin(angle), zero_vec,
                                                  torch.sin(angle), torch.cos(angle), zero_vec,
                                                  zero_vec, zero_vec, one_vec])
    return trafo_matrix

def canonical_trafo(coords_xyz):
    """
    Transforms the given real xyz coordinates into a canonical frame using PyTorch.
    
    This transformation aligns the keypoints of hands into a canonical frame where the palm's root keypoint is 
    at the origin, the root bone is aligned along the y-axis, and a specified keypoint (e.g., the beginning of 
    the pinky) is positioned such that it defines a rotation around the y-axis. This alignment is intended to 
    help the network learn reasonable shape priors by ensuring that the hands across all frames are nicely aligned.
    
    Inputs:
        coords_xyz: A tensor of shape BxNx3, where B is the batch size, N is the number of keypoints per hand 
                    (fixed at 21 for this function), and 3 represents the xyz coordinates in 3D space.
    
    Returns:
        coords_xyz_normed: A tensor of shape BxNx3, representing the transformed coordinates of each keypoint
                           in the canonical frame for each sample in the batch. This maintains the original 
                           batch size and number of keypoints, but the coordinates have been transformed.
        
        total_rot_mat: A tensor of shape Bx3x3, representing the total rotation matrix applied to transform 
                       the coordinates from the original frame to the canonical frame for each sample in the batch.
                       This matrix can be used to understand the rotation applied during the transformation process.
    
    The function performs the transformation through a series of steps:
    1. Translation of all keypoints such that the palm's root keypoint is at the origin.
    2. Rotation and scaling of keypoints to align the root bone along the y-axis.
    3. Additional rotation to ensure a specified keypoint defines the orientation around the y-axis.
    
    This systematic approach ensures a consistent and normalized representation of hand keypoints across different frames,
    facilitating the learning process in tasks related to hand pose estimation.
    """
    # Function implementation here...

    coords_xyz = coords_xyz.view(-1, 21, 3)

    ROOT_NODE_ID = 0  # Node that will be at 0/0/0: 0=palm keypoint (root)
    ALIGN_NODE_ID = 12  # Node that will be at 0/-D/0: 12=beginning of middle finger
    ROT_NODE_ID = 20  # Node that will be at z=0, x>0; 20: Beginning of pinky

    # 1. Translate the whole set so that the root kp is located in the origin
    trans = coords_xyz[:, ROOT_NODE_ID, :].unsqueeze(1)
    coords_xyz_t = coords_xyz - trans

    # 2. Rotate and scale keypoints
    p = coords_xyz_t[:, ALIGN_NODE_ID, :]  # The point we want to put on (0/1/0)

    # Rotate point into the yz-plane
    alpha = atan2_pytorch(p[:, 0], p[:, 1])
    rot_mat = _get_rot_mat_z(alpha)
    coords_xyz_t_r1 = torch.matmul(coords_xyz_t, rot_mat.permute(0, 2, 1))
    total_rot_mat = rot_mat

    # Rotate point within the yz-plane onto the xy-plane
    p = coords_xyz_t_r1[:, ALIGN_NODE_ID, :]
    beta = -atan2_pytorch(p[:, 2], p[:, 1])
    rot_mat = _get_rot_mat_x(beta + 3.141592653589793)
    coords_xyz_t_r2 = torch.matmul(coords_xyz_t_r1, rot_mat.permute(0, 2, 1))
    total_rot_mat = torch.matmul(total_rot_mat, rot_mat)

    # 3. Rotate keypoints such that rotation along the y-axis is defined
    p = coords_xyz_t_r2[:, ROT_NODE_ID, :]
    gamma = atan2_pytorch(p[:, 2], p[:, 0])
    rot_mat = _get_rot_mat_y(gamma)
    coords_xyz_normed = torch.matmul(coords_xyz_t_r2, rot_mat.permute(0, 2, 1))
    total_rot_mat = torch.matmul(total_rot_mat, rot_mat)

    return coords_xyz_normed, total_rot_mat



def flip_right_hand(coords_xyz_canonical, cond_right):
    """Flips the given canonical coordinates in PyTorch, when cond_right is true. 
    Returns coords unchanged otherwise. The returned coordinates represent those of a left hand.

    Inputs:
        coords_xyz_canonical: Nx3 tensor, containing the coordinates for each of the N keypoints
    """
    expanded = False
    if coords_xyz_canonical.dim() == 2:
        coords_xyz_canonical = coords_xyz_canonical.unsqueeze(0)
        cond_right = cond_right.unsqueeze(0)
        expanded = True

    # mirror along y axis
    coords_xyz_canonical_mirrored = torch.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], -coords_xyz_canonical[:, :, 2]], dim=-1)

    # select mirrored in case it was a right hand
    coords_xyz_canonical_left = torch.where(cond_right.unsqueeze(-1), coords_xyz_canonical_mirrored, coords_xyz_canonical)

    if expanded:
        coords_xyz_canonical_left = coords_xyz_canonical_left.squeeze(0)

    return coords_xyz_canonical_left

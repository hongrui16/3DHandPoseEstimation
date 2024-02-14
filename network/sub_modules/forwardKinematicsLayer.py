import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import math

import sys
sys.path.append('../..')  

from config.config import *

    

def get_right_hand_rotation_matrix(angles, device = 'cpu'):
    """ Compute the rotation matrix based on the provided angle vector.
    Angles vector contains rotation angles in radians along the x, y, and z axis.
    """
    sin, cos = torch.sin, torch.cos
    x, y, z = angles
    R_x = torch.tensor([[1, 0, 0],
                    [0, cos(x), -sin(x)],
                    [0, sin(x), cos(x)]], device=device, dtype=torch.float32)

    R_y = torch.tensor([[cos(y), 0, sin(y)],
                    [0, 1, 0],
                    [-sin(y), 0, cos(y)]], device=device, dtype=torch.float32)
    R_z = torch.tensor([[cos(z), -sin(z), 0],
                    [sin(z), cos(z), 0],
                    [0, 0, 1]], device=device, dtype=torch.float32)
        
    
    R = R_x @ R_y @ R_z
    return R

def get_left_hand_rotation_matrix(angles, device = 'cpu'):
    """ Compute the rotation matrix for a left-handed coordinate system
    based on the provided angle vector. Angles vector contains rotation
    angles in radians along the x, y, and z axis.
    """
    sin, cos = torch.sin, torch.cos
    x, y, z = angles
    R_x = torch.tensor([[1, 0, 0],
                        [0, cos(x), sin(x)],
                        [0, -sin(x), cos(x)]], device=device, dtype=torch.float32)

    R_y = torch.tensor([[cos(y), 0, -sin(y)],
                        [0, 1, 0],
                        [sin(y), 0, cos(y)]], device=device, dtype=torch.float32)

    R_z = torch.tensor([[cos(z), sin(z), 0],
                        [-sin(z), cos(z), 0],
                        [0, 0, 1]], device=device, dtype=torch.float32)

    R = R_x @ R_y @ R_z
    return R


def get_right_hand_batch_rotation_matrix(angles, device='cpu'):

    """ 
    angles size: B*3 (B: batch_size; 3: number of rotation angles in radian for a joint)
    Compute the rotation matrix based on the provided angle tensor.
    angles tensor contains rotation angles in radians along the x, y, and z axis for multiple samples in a batch.
    """
    sin, cos = torch.sin, torch.cos
    
    batch_size = angles.size(0)
    # print(f'angles.shape: {angles.shape}')
    # Initialize empty rotation matrices
    R_x = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    R_y = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    R_z = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)

    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]

    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos(x)
    R_x[:, 1, 2] = -sin(x)
    R_x[:, 2, 1] = sin(x)
    R_x[:, 2, 2] = cos(x)

    R_y[:, 0, 0] = cos(y)
    R_y[:, 0, 2] = sin(y)
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin(y)
    R_y[:, 2, 2] = cos(y)
    
    R_z[:, 0, 0] = cos(z)
    R_z[:, 0, 1] = -sin(z)
    R_z[:, 1, 0] = sin(z)
    R_z[:, 1, 1] = cos(z)
    R_z[:, 2, 2] = 1
    
    R = torch.bmm(torch.bmm(R_x, R_y), R_z)
    return R




def get_left_hand_batch_rotation_matrix(angles, device='cpu'):

    """ 
    angles size: B*3 (B: batch_size; 3: number of rotation angles in radian for a joint)
    Compute the rotation matrix based on the provided angle tensor.
    angles tensor contains rotation angles in radians along the x, y, and z axis for multiple samples in a batch.
    """
    sin, cos = torch.sin, torch.cos
    
    batch_size = angles.size(0)
    # print(f'angles.shape: {angles.shape}')
    # Initialize empty rotation matrices
    R_x = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    R_y = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    R_z = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)

    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]

    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos(x)
    R_x[:, 1, 2] = sin(x)
    R_x[:, 2, 1] = -sin(x)
    R_x[:, 2, 2] = cos(x)

    R_y[:, 0, 0] = cos(y)
    R_y[:, 0, 2] = -sin(y)
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = sin(y)
    R_y[:, 2, 2] = cos(y)
    
    R_z[:, 0, 0] = cos(z)
    R_z[:, 0, 1] = sin(z)
    R_z[:, 1, 0] = -sin(z)
    R_z[:, 1, 1] = cos(z)
    R_z[:, 2, 2] = 1
    
    R = torch.bmm(torch.bmm(R_x, R_y), R_z)
    return R



class ForwardKinematics(nn.Module):
    def __init__(self, device = 'cpu'):
        super(ForwardKinematics, self).__init__()
        self.device = device
    
    def forward(self, root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root):
        '''
        right hand coordinate system, all angles are radian.
        X-axis: Points in the direction of your thumb, horizontally to the right.
        Y-axis: perpendicular to the direction of the palm, upward.
        Z-axis: Points in the direction of the tip of the middle finger, forward.

        
        If the middle finger is bent 10 degrees downward, it is a rotation around the X-axis
        The expansion/merging of the index finger (Abduction/Adduction) is performed around the Y-axis.
        If the entire palm rotates about the direction of the tip of the middle finger (i.e. the Z-axis), then this is a rotation about the Z-axis.

        nodes = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4']

                wrist(root)
                    |
        --------------------------
        A1    B1    C1    D1    E1
        |     |     |     |     |
        v     v     v     v     v
        A2    B2    C2    D2    E2
        |     |     |     |     |
        v     v     v     v     v
        A3    B3    C3    D3    E3
        |     |     |     |     |
        v     v     v     v     v
        A4    B4    C4    D4    E4

        A: thumb
        B：index
        C：middle 
        D：ring 
        E: pinky
        
        root_angles: 
            size: B*3
            three angles of root (previous MLP prediction, x y z direction rotation angle)
        other_angles: 
            size: B*23
            There are 23 other_angles in total, from *1 to *3, * represents A, B, C, D, E, the order is from 1~3, from A~E.
            
            ## thumb
            other_angles[0:3] is the three rotation angles of A1 in the x y z direction, A1, 
            other_angles[3:6] is the three rotation angles of A2 in the x y z direction, A2, 
            other_angles[6] is a rotation angle in the Y direction of A3, 

            ## other fingers
            other_angles[7:9] is the two rotation angles of B1 in the x and Y directions
            other_angles[9] is a rotation angle in the x direction of B2
            other_angles[10] is a rotation angle in the x direction of B3
            C/D/E, all *1 have two rotation angles in the x and z directions, *2, *3 only have one rotation angle in the x direction

        bone_lengths:
            size: B*20
            There are 20 bone_length in total, which is the edge length of the graph, from root-A1, A1-A2, A2-A3, A3-A4, root-B1, B1-B2 in this order
        '''

        # Check if camera_intrinsic_matrix is a Tensor
        assert isinstance(camera_intrinsic_matrix, torch.Tensor)
            # # If it is not a Tensor, convert it to a Tensor and put it on the device
            # camera_intrinsic_matrix = torch.tensor(camera_intrinsic_matrix, dtype=torch.float32, device=self.device)

        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        # print(f"root_angles is on device: {root_angles.device}") #device: cpu

        bs, num_points = bone_lengths.shape       
        positions_xyz = torch.zeros(bs, 1, 3, device=self.device)  # root xyz position
        rotations = get_right_hand_batch_rotation_matrix(root_angles, self.device) 
        # print(f'rotations.shape', rotations.shape) # # torch.Size([bs, 3, 3])
        rotations = rotations.unsqueeze(dim=1)
        # print(f'rotations.shape', rotations.shape) # # torch.Size([bs, 1, 3, 3])

        # Define the sequence of nodes
        nodes = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4']
        angle_idx = 0
        # print('bone_lengths.shape', bone_lengths.shape)
        for i, node in enumerate(nodes):
            if i % 4 == 0:  # First joint of each finger
                parent_rotation = rotations[:, 0]
                parent_position = positions_xyz[:, 0]
            else:
                parent_rotation = rotations[:, i]
                parent_position = positions_xyz[:, i]
            
            # Get the current bone length
            bone_length = bone_lengths[:, i]

            # print(f"other_angles is on device: {other_angles.device}") #device: cpu
            # print(f"parent_rotation is on device: {parent_rotation.device}") #device: cpu
            # print(f"positions_xyz is on device: {positions_xyz.device}") #device: cpu

            if node.startswith('A'):
                ## finger = 'thumb'
                # Get the rotation for the current node
                if node.endswith('1') or node.endswith('2'):  # First joint of each finger, has x, y, z rotation angles
                    joint_angles = other_angles[:, angle_idx:angle_idx+3]
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 3                
                elif node.endswith('3'):  # Second and third joint of each finger, has only Y rotation angle
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 1] = other_angles[:, angle_idx]
                    # print(f'joint_angles.shape: {joint_angles.shape}') ##  torch.Size([bs, 3])
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 1
                else:
                    # elif node.endswith('4'):  # Tip of the finger, has no rotation of its own
                    joint_local_rotation = torch.stack([torch.eye(3, device=self.device) for _ in range(bs)])
                    # print(f'joint_local_rotation.shape: {joint_local_rotation.shape}') ##  torch.Size([bs, 3, 3])

            else:
                ## finger = 'others'
                # Get the rotation for the current node
                if node.endswith('1'):  # First joint of each finger, has x, y rotation angles
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 0] = other_angles[:, angle_idx] # x angles
                    joint_angles[:, 1] = other_angles[:, angle_idx+1] # Y angles
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 2
                
                elif not node.endswith('4'):  # Second and third joint of each finger, has only x rotation angle
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 0] = other_angles[:, angle_idx]
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 1
                else:
                    # elif node.endswith('4'):  # Tip of the finger, has no rotation of its own
                    joint_local_rotation = torch.stack([torch.eye(3, device=self.device) for _ in range(bs)])
            # print(f'joint_local_rotation.shape', joint_local_rotation.shape) # torch.Size([bs, 3，3])
            # print(f'node: {node}')
            # print(f'joint_local_rotation: {joint_local_rotation}')
            # print(f'parent_rotation: {parent_rotation}')
            # print(f'parent_rotation.shape', parent_rotation.shape) # torch.Size([bs, 3, 3])
            # print(f'joint_local_rotation.shape', joint_local_rotation.shape) # torch.Size([bs, 3, 3])
                    
            # print(f"parent_rotation is on device: {parent_rotation.device}") #device: cpu
            # print(f"joint_local_rotation is on device: {joint_local_rotation.device}") #device: cpu

            # Calculate the global rotation for the current node
            joint_global_rotation = parent_rotation @ joint_local_rotation
            # print(f'joint_global_rotation.shape', joint_global_rotation.shape) # torch.Size([bs, 3, 3])
            # print('-')
            #### Calculate the global position for the current node
            joint_offset = torch.zeros(bs, 3, device=self.device)
            # print(f'bone_length.shape', bone_length.shape) # torch.Size([bs])
            joint_offset[:, 2] = bone_length
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 3])

            joint_offset = joint_offset.unsqueeze(1)
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 1, 3])

            joint_offset = joint_offset.transpose(1, 2)
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 3, 1])

            # print(f"joint_global_rotation is on device: {joint_global_rotation.device}") #device: cpu
            # print(f"joint_offset is on device: {joint_offset.device}") #cuda:0

            # Calculate the global position for the current node using batch matrix multiplication
            offset = torch.bmm(joint_global_rotation, joint_offset)
            # print(f'offset.shape', offset.shape) # torch.Size([bs, 3, 1])

            global_position = parent_position + offset.squeeze()
            # print(f'global_position.shape', global_position.shape) # torch.Size([bs, 3])


            # Store the global rotation and position
            rotations = torch.cat([rotations, joint_global_rotation.unsqueeze(1)], dim=1)
            # print(f'rotations.shape', rotations.shape) # torch.Size([bs, num_joint, 3, 3])

            positions_xyz = torch.cat([positions_xyz, global_position.unsqueeze(1)], dim=1)
            # print(f'positions_xyz.shape', positions_xyz.shape) # torch.Size([bs, num_joint, 3])


        # print(f'positions_xyz.shape', positions_xyz.shape) # torch.Size([bs, 21, 3])
        kp_coord_xyz21_absolute = self.convert_rel_normalized_to_absolute(positions_xyz, index_root_bone_length, kp_coord_xyz_root)
        # print(f'kp_coord_xyz21_absolute.shape', kp_coord_xyz21_absolute.shape) # torch.Size([bs, 21, 3])


        for i in range(1, 21, 4):
            kp_coord_xyz21_absolute[:,[i, i + 3]] = kp_coord_xyz21_absolute[:,[i + 3, i]]
            kp_coord_xyz21_absolute[:,[i + 1, i + 2]] = kp_coord_xyz21_absolute[:,[i + 2, i + 1]]

        kp_coord_uv21 = self.project_xyz_to_uv(kp_coord_xyz21_absolute, self.camera_intrinsic_matrix)
        return [kp_coord_xyz21_absolute, kp_coord_uv21]
    

    def convert_rel_normalized_to_absolute(self, kp_coord_xyz21_rel_normed, index_root_bone_length, kp_coord_xyz_root):
        """
        Args:
            kp_coord_xyz21_rel_normed: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            index_root_bone_length,  shape (batch_size, 1).
            kp_coord_xyz_root,  shape (batch_size, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # Zoom positions_xyz
        index_root_bone_length_expanded = index_root_bone_length.unsqueeze(-1) # [batch_size, 1, 1]
        positions_xyz_scaled = kp_coord_xyz21_rel_normed * index_root_bone_length_expanded # [batch_size, num_points, 3]

        # Adjust the shape of kp_coord_xyz_root to facilitate addition operations
        kp_coord_xyz_root_expanded = kp_coord_xyz_root.unsqueeze(1) # [batch_size, 1, 3]

        # Add scaled positions_xyz_scaled to kp_coord_xyz_root
        positions_xyz_absloute = positions_xyz_scaled + kp_coord_xyz_root_expanded # [batch_size, num_points, 3]

        # If necessary, dimension exchange can be performed here
        # positions_xyz_final = positions_xyz_abslute.permute(0, 2, 1) # [batch_size, 3, num_points]
        return positions_xyz_absloute

    def project(self, positions_xyz, camera_intrinsic_matrix, index_root_bone_length):
        """
        Projects three-dimensional coordinates onto the image plane.

        Args:
            positions_xyz: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            camera_intrinsic_matrix: Camera intrinsic parameter matrix, shape is (batch_size, 3, 3).
            index_root_bone_length,  shape (batch_size, 1).
            kp_coord_xyz_root,  shape (batch_size, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # print(f'positions_xyz.shape: {positions_xyz.shape}') #torch.Size([2, 21, 3]
        # print(f'camera_intrinsic_matrix.shape: {camera_intrinsic_matrix.shape}') #torch.Size([2, 21, 3]
        # print('positions_xyz[0]', positions_xyz[0])
        

        # Adjust the shape of positions_xyz to match camera_intrinsic_matrix
        # Reshape to (bs, 3, num_points)
        points_3d_reshaped = positions_xyz.permute(0, 2, 1)
        # points_3d_reshaped shape is [bs, 3, num_points]

        index_root_bone_length_expanded = index_root_bone_length.unsqueeze(2)
        points_3d_reshaped = points_3d_reshaped * index_root_bone_length_expanded


        # Use batch matrix multiplication
        # camera_intrinsic_matrix shape is [bs, 3, 3]
        p = torch.bmm(camera_intrinsic_matrix, points_3d_reshaped)

        # The shape of p should now be [bs, 3, num_points]

        # Check if the last row of p has any zero values and replace with a small non-zero value to avoid dividing by zero
        p[:, -1, :] = torch.where(p[:, -1, :] == 0, torch.tensor(1e-10, dtype=p.dtype, device=p.device), p[ :, -1, :])
        
        # print(f'p.shape: {p.shape}') # should be [bs, num_points, 2]

        #Normalize to get final 2D coordinates
        #The shape becomes [bs, num_points, 2]
        uv = (p[:, :2, :] / p[:, -1, :].unsqueeze(1)).permute(0, 2, 1)

        # print(f'uv.shape: {uv.shape}') # should be [bs, num_points, 2]

        # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
        # uv = uv.t().view(bs, num_points, 2)
        # print('uv[0]', uv[0])

        return uv

    def project_xyz_to_uv(self, positions_xyz, camera_intrinsic_matrix):
        """
        Projects three-dimensional coordinates onto the image plane.

        Args:
            positions_xyz: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            camera_intrinsic_matrix: Camera intrinsic parameter matrix, shape is (batch_size, 3, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # print(f'positions_xyz.shape: {positions_xyz.shape}') #torch.Size([2, 21, 3]
        # print(f'camera_intrinsic_matrix.shape: {camera_intrinsic_matrix.shape}') #torch.Size([2, 21, 3]
        # print('positions_xyz[0]', positions_xyz[0])
        

        # bs, num_points, _ = positions_xyz.shape

        # Adjust the shape of positions_xyz to match camera_intrinsic_matrix
        # Reshape to (bs, 3, num_points)
        points_3d_reshaped = positions_xyz.permute(0, 2, 1)
        # points_3d_reshaped shape is [bs, 3, num_points]


        # Use batch matrix multiplication
        # camera_intrinsic_matrix shape is [bs, 3, 3]
        p = torch.bmm(camera_intrinsic_matrix, points_3d_reshaped)

        # The shape of p should now be [bs, 3, num_points]

        # Check if the last row of p has any zero values and replace with a small non-zero value to avoid dividing by zero
        p[:, -1, :] = torch.where(p[:, -1, :] == 0, torch.tensor(1e-10, dtype=p.dtype, device=p.device), p[ :, -1, :])
        
        # print(f'p.shape: {p.shape}') # should be [bs, num_points, 2]

        #Normalize to get final 2D coordinates
        #The shape becomes [bs, num_points, 2]
        uv = (p[:, :2, :] / p[:, -1, :].unsqueeze(1)).permute(0, 2, 1)

        # print(f'uv.shape: {uv.shape}') # should be [bs, num_points, 2]

        # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
        # uv = uv.t().view(bs, num_points, 2)
        # print('uv[0]', uv[0])

        return uv



class ForwardKinematicsMatchGTOrder(nn.Module):
    def __init__(self, device = 'cpu'):
        super(ForwardKinematics, self).__init__()
        self.device = device
    
    

    def forward(self, root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root):
        '''
        right hand coordinate system, all angles are radian.
        X-axis: Points in the direction of your thumb, horizontally to the right.
        Y-axis: perpendicular to the direction of the palm, upward.
        Z-axis: Points in the direction of the tip of the middle finger, forward.

        
        If the middle finger is bent 10 degrees downward, it is a rotation around the X-axis
        The expansion/merging of the index finger (Abduction/Adduction) is performed around the Y-axis.
        If the entire palm rotates about the direction of the tip of the middle finger (i.e. the Z-axis), then this is a rotation about the Z-axis.

        nodes = ['A4', 'A3', 'A2', 'A1', 'B4', 'B3', 'B2', 'B1', 'C4', 'C3', 'C2', 'C1', 'D4', 'D3', 'D2', 'D1', 'E4', 'E3', 'E2', 'E1']

                wrist(root)
                    |
        --------------------------
        A1    B1    C1    D1    E1
        |     |     |     |     |
        v     v     v     v     v
        A2    B2    C2    D2    E2
        |     |     |     |     |
        v     v     v     v     v
        A3    B3    C3    D3    E3
        |     |     |     |     |
        v     v     v     v     v
        A4    B4    C4    D4    E4


        A: thumb
        B：index
        C：middle 
        D：ring 
        E: pinky
        
        root_angles: 
            size: B*3
            three angles of root (previous MLP prediction, x y z direction rotation angle)
        other_angles: 
            size: B*23
            There are 23 other_angles in total, from *1 to *3, * represents A, B, C, D, E, the order is from 1~3, from A~E.
            
            ## thumb
            other_angles[0:3] is the three rotation angles of A1 in the x y z direction, A1, 
            other_angles[3:6] is the three rotation angles of A2 in the x y z direction, A2, 
            other_angles[6] is a rotation angle in the Y direction of A3, 

            ## other fingers
            other_angles[7:9] is the two rotation angles of B1 in the x and Y directions
            other_angles[9] is a rotation angle in the x direction of B2
            other_angles[10] is a rotation angle in the x direction of B3
            C/D/E, all *1 have two rotation angles in the x and z directions, *2, *3 only have one rotation angle in the x direction

        bone_lengths:
            size: B*20
            There are 20 bone_length in total, which is the edge length of the graph, from root-A1, A1-A2, A2-A3, A3-A4, root-B1, B1-B2 in this order
        '''

        # Check if camera_intrinsic_matrix is a Tensor
        assert isinstance(camera_intrinsic_matrix, torch.Tensor)
            # # If it is not a Tensor, convert it to a Tensor and put it on the device
            # camera_intrinsic_matrix = torch.tensor(camera_intrinsic_matrix, dtype=torch.float32, device=self.device)

        self.camera_intrinsic_matrix = camera_intrinsic_matrix
        # print(f"root_angles is on device: {root_angles.device}") #device: cpu

        bs, num_points = bone_lengths.shape       
        positions_xyz = torch.zeros(bs, 1, 3, device=self.device)  # root xyz position
        rotations = get_right_hand_batch_rotation_matrix(root_angles, self.device) 
        # print(f'rotations.shape', rotations.shape) # # torch.Size([bs, 3, 3])
        rotations = rotations.unsqueeze(dim=1)
        # print(f'rotations.shape', rotations.shape) # # torch.Size([bs, 1, 3, 3])

        # Define the sequence of nodes
        nodes = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4']
        angle_idx = 0
        # print('bone_lengths.shape', bone_lengths.shape)
        for i, node in enumerate(nodes):
            if i % 4 == 0:  # First joint of each finger
                parent_rotation = rotations[:, 0]
                parent_position = positions_xyz[:, 0]
            else:
                parent_rotation = rotations[:, i]
                parent_position = positions_xyz[:, i]
            
            # Get the current bone length
            bone_length = bone_lengths[:, i]

            # print(f"other_angles is on device: {other_angles.device}") #device: cpu
            # print(f"parent_rotation is on device: {parent_rotation.device}") #device: cpu
            # print(f"positions_xyz is on device: {positions_xyz.device}") #device: cpu

            if node.startswith('A'):
                ## finger = 'thumb'
                # Get the rotation for the current node
                if node.endswith('1') or node.endswith('2'):  # First joint of each finger, has x, y, z rotation angles
                    joint_angles = other_angles[:, angle_idx:angle_idx+3]
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 3                
                elif node.endswith('3'):  # Second and third joint of each finger, has only Y rotation angle
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 1] = other_angles[:, angle_idx]
                    # print(f'joint_angles.shape: {joint_angles.shape}') ##  torch.Size([bs, 3])
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 1
                else:
                    # elif node.endswith('4'):  # Tip of the finger, has no rotation of its own
                    joint_local_rotation = torch.stack([torch.eye(3, device=self.device) for _ in range(bs)])
                    # print(f'joint_local_rotation.shape: {joint_local_rotation.shape}') ##  torch.Size([bs, 3, 3])

            else:
                ## finger = 'others'
                # Get the rotation for the current node
                if node.endswith('1'):  # First joint of each finger, has x, y rotation angles
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 0] = other_angles[:, angle_idx] # x angles
                    joint_angles[:, 1] = other_angles[:, angle_idx+1] # Y angles
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 2
                
                elif not node.endswith('4'):  # Second and third joint of each finger, has only x rotation angle
                    joint_angles = torch.zeros((joint_angles.shape[0], 3), device=self.device)
                    joint_angles[:, 0] = other_angles[:, angle_idx]
                    joint_local_rotation = get_right_hand_batch_rotation_matrix(joint_angles, self.device)
                    angle_idx += 1
                else:
                    # elif node.endswith('4'):  # Tip of the finger, has no rotation of its own
                    joint_local_rotation = torch.stack([torch.eye(3, device=self.device) for _ in range(bs)])
            # print(f'joint_local_rotation.shape', joint_local_rotation.shape) # torch.Size([bs, 3，3])
            # print(f'node: {node}')
            # print(f'joint_local_rotation: {joint_local_rotation}')
            # print(f'parent_rotation: {parent_rotation}')
            # print(f'parent_rotation.shape', parent_rotation.shape) # torch.Size([bs, 3, 3])
            # print(f'joint_local_rotation.shape', joint_local_rotation.shape) # torch.Size([bs, 3, 3])
                    
            # print(f"parent_rotation is on device: {parent_rotation.device}") #device: cpu
            # print(f"joint_local_rotation is on device: {joint_local_rotation.device}") #device: cpu

            # Calculate the global rotation for the current node
            joint_global_rotation = parent_rotation @ joint_local_rotation
            # print(f'joint_global_rotation.shape', joint_global_rotation.shape) # torch.Size([bs, 3, 3])
            # print('-')
            #### Calculate the global position for the current node
            joint_offset = torch.zeros(bs, 3, device=self.device)
            # print(f'bone_length.shape', bone_length.shape) # torch.Size([bs])
            joint_offset[:, 2] = bone_length
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 3])

            joint_offset = joint_offset.unsqueeze(1)
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 1, 3])

            joint_offset = joint_offset.transpose(1, 2)
            # print(f'joint_offset.shape', joint_offset.shape) # torch.Size([bs, 3, 1])

            # print(f"joint_global_rotation is on device: {joint_global_rotation.device}") #device: cpu
            # print(f"joint_offset is on device: {joint_offset.device}") #cuda:0

            # Calculate the global position for the current node using batch matrix multiplication
            offset = torch.bmm(joint_global_rotation, joint_offset)
            # print(f'offset.shape', offset.shape) # torch.Size([bs, 3, 1])

            global_position = parent_position + offset.squeeze()
            # print(f'global_position.shape', global_position.shape) # torch.Size([bs, 3])


            # Store the global rotation and position
            rotations = torch.cat([rotations, joint_global_rotation.unsqueeze(1)], dim=1)
            # print(f'rotations.shape', rotations.shape) # torch.Size([bs, num_joint, 3, 3])

            positions_xyz = torch.cat([positions_xyz, global_position.unsqueeze(1)], dim=1)
            # print(f'positions_xyz.shape', positions_xyz.shape) # torch.Size([bs, num_joint, 3])


        # print(f'positions_xyz.shape', positions_xyz.shape) # torch.Size([bs, 21, 3])
        kp_coord_xyz21_absolute = self.convert_rel_normalized_to_absolute(positions_xyz, index_root_bone_length, kp_coord_xyz_root)
        # print(f'kp_coord_xyz21_absolute.shape', kp_coord_xyz21_absolute.shape) # torch.Size([bs, 21, 3])


        for i in range(1, 21, 4):
            kp_coord_xyz21_absolute[:,[i, i + 3]] = kp_coord_xyz21_absolute[:,[i + 3, i]]
            kp_coord_xyz21_absolute[:,[i + 1, i + 2]] = kp_coord_xyz21_absolute[:,[i + 2, i + 1]]

        kp_coord_uv21 = self.project_xyz_to_uv(kp_coord_xyz21_absolute, self.camera_intrinsic_matrix)
        return [kp_coord_xyz21_absolute, kp_coord_uv21]

    def convert_rel_normalized_to_absolute(self, kp_coord_xyz21_rel_normed, index_root_bone_length, kp_coord_xyz_root):
        """
        Args:
            kp_coord_xyz21_rel_normed: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            index_root_bone_length,  shape (batch_size, 1).
            kp_coord_xyz_root,  shape (batch_size, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # Zoom positions_xyz
        index_root_bone_length_expanded = index_root_bone_length.unsqueeze(-1) # [batch_size, 1, 1]
        positions_xyz_scaled = kp_coord_xyz21_rel_normed * index_root_bone_length_expanded # [batch_size, num_points, 3]

        # Adjust the shape of kp_coord_xyz_root to facilitate addition operations
        kp_coord_xyz_root_expanded = kp_coord_xyz_root.unsqueeze(1) # [batch_size, 1, 3]

        # Add scaled positions_xyz_scaled to kp_coord_xyz_root
        positions_xyz_absloute = positions_xyz_scaled + kp_coord_xyz_root_expanded # [batch_size, num_points, 3]

        # If necessary, dimension exchange can be performed here
        # positions_xyz_final = positions_xyz_abslute.permute(0, 2, 1) # [batch_size, 3, num_points]
        return positions_xyz_absloute

    def project(self, positions_xyz, camera_intrinsic_matrix, index_root_bone_length):
        """
        Projects three-dimensional coordinates onto the image plane.

        Args:
            positions_xyz: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            camera_intrinsic_matrix: Camera intrinsic parameter matrix, shape is (batch_size, 3, 3).
            index_root_bone_length,  shape (batch_size, 1).
            kp_coord_xyz_root,  shape (batch_size, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # print(f'positions_xyz.shape: {positions_xyz.shape}') #torch.Size([2, 21, 3]
        # print(f'camera_intrinsic_matrix.shape: {camera_intrinsic_matrix.shape}') #torch.Size([2, 21, 3]
        # print('positions_xyz[0]', positions_xyz[0])
        

        # Adjust the shape of positions_xyz to match camera_intrinsic_matrix
        # Reshape to (bs, 3, num_points)
        points_3d_reshaped = positions_xyz.permute(0, 2, 1)
        # points_3d_reshaped shape is [bs, 3, num_points]

        index_root_bone_length_expanded = index_root_bone_length.unsqueeze(2)
        points_3d_reshaped = points_3d_reshaped * index_root_bone_length_expanded


        # Use batch matrix multiplication
        # camera_intrinsic_matrix shape is [bs, 3, 3]
        p = torch.bmm(camera_intrinsic_matrix, points_3d_reshaped)

        # The shape of p should now be [bs, 3, num_points]

        # Check if the last row of p has any zero values and replace with a small non-zero value to avoid dividing by zero
        p[:, -1, :] = torch.where(p[:, -1, :] == 0, torch.tensor(1e-10, dtype=p.dtype, device=p.device), p[ :, -1, :])
        
        # print(f'p.shape: {p.shape}') # should be [bs, num_points, 2]

        #Normalize to get final 2D coordinates
        #The shape becomes [bs, num_points, 2]
        uv = (p[:, :2, :] / p[:, -1, :].unsqueeze(1)).permute(0, 2, 1)

        # print(f'uv.shape: {uv.shape}') # should be [bs, num_points, 2]

        # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
        # uv = uv.t().view(bs, num_points, 2)
        # print('uv[0]', uv[0])

        return uv

    def project_xyz_to_uv(self, positions_xyz, camera_intrinsic_matrix):
        """
        Projects three-dimensional coordinates onto the image plane.

        Args:
            positions_xyz: Three-dimensional coordinates, shape (batch_size, num_points, 3).
            camera_intrinsic_matrix: Camera intrinsic parameter matrix, shape is (batch_size, 3, 3).
        Returns:
            The projected two-dimensional UV coordinates, shape (batch_size, num_points, 2).
        """
        # print(f'positions_xyz.shape: {positions_xyz.shape}') #torch.Size([2, 21, 3]
        # print(f'camera_intrinsic_matrix.shape: {camera_intrinsic_matrix.shape}') #torch.Size([2, 21, 3]
        # print('positions_xyz[0]', positions_xyz[0])
        

        # bs, num_points, _ = positions_xyz.shape

        # Adjust the shape of positions_xyz to match camera_intrinsic_matrix
        # Reshape to (bs, 3, num_points)
        points_3d_reshaped = positions_xyz.permute(0, 2, 1)
        # points_3d_reshaped shape is [bs, 3, num_points]


        # Use batch matrix multiplication
        # camera_intrinsic_matrix shape is [bs, 3, 3]
        p = torch.bmm(camera_intrinsic_matrix, points_3d_reshaped)

        # The shape of p should now be [bs, 3, num_points]

        # Check if the last row of p has any zero values and replace with a small non-zero value to avoid dividing by zero
        p[:, -1, :] = torch.where(p[:, -1, :] == 0, torch.tensor(1e-10, dtype=p.dtype, device=p.device), p[ :, -1, :])
        
        # print(f'p.shape: {p.shape}') # should be [bs, num_points, 2]

        #Normalize to get final 2D coordinates
        #The shape becomes [bs, num_points, 2]
        uv = (p[:, :2, :] / p[:, -1, :].unsqueeze(1)).permute(0, 2, 1)

        # print(f'uv.shape: {uv.shape}') # should be [bs, num_points, 2]

        # Convert the shape of uv from (2, bs*num) to (bs, num, 2)
        # uv = uv.t().view(bs, num_points, 2)
        # print('uv[0]', uv[0])

        return uv






if __name__  == '__main__':
    # sys.path.append('../..')  
    # Define your root_angles, other_angles, and bone_lengths
    bs = 1
    root_angles = torch.rand((1, 3))
    other_angles = torch.rand(1, 23)  # Replace with actual values
    bone_lengths = torch.rand(1, 20)  # Replace with actual values

    # print('root_angles', root_angles)

    # print(f'other_angles[0,0:6]: {other_angles[0,0:6]}')
    # print(f'bone_lengths[0,0:3]: {bone_lengths[0,0:3]}')


    camera_intrinsic_matrix = torch.tensor(
    [[[282.9000,   0.0000, 160.0000],
         [  0.0000, 282.9000, 160.0000],
         [  0.0000,   0.0000,   1.0000]]], dtype=torch.float32)

    forward_kinematics = ForwardKinematics()

    # Calculate the positions
    # positions = forward_kinematics(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix)
    # positions_xyz, positions_uv = positions
    # # Convert positions to a more readable format if needed, e.g., a list of tuples
    # print('positions_xyz.shape', positions_xyz.shape)
    # print(f'positions_xyz[0,0:3]: {positions_xyz[0,0:3]}')
    keypoint_xyz21_normed = torch.tensor(
        [
            [
                [ 0.0000,  0.0000,  1.0000],
                [ 1,  2, 2],
                [ -1,  2, 4],
                [ 1,  2, 1],
                # [ 0.3935,  0.5777, -0.8612],
                # [-0.6272,  2.1604, -4.3848],
                # [-0.4883,  1.8418, -3.9199],
                # [-0.3185,  1.3671, -3.2746],
                # [-0.0822,  0.9387, -2.5328],
                # [-0.7223,  3.0409, -3.4931],
                # [-0.7756,  2.3451, -3.3204],
                # [-0.8046,  1.6778, -3.1375],
                # [-0.6458,  1.0149, -2.4058],
                # [-1.0258,  2.9434, -3.1171],
                # [-1.0845,  2.2369, -2.8580],
                # [-1.1188,  1.6515, -2.6853],
                # [-1.0037,  1.0606, -2.0857],
                # [-1.6457,  2.6677, -2.6649],
                # [-1.6510,  2.2775, -2.4947],
                # [-1.6282,  1.8484, -2.3296],
                # [-1.3525,  1.1717, -1.5954]
            ]
        ])
    
    keypoint_xyz21_rel_normed = torch.tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.2332,  2.2252, -3.0206],
         [ 0.3983,  1.7986, -2.4160]]])
    
    keypoint_scale = torch.tensor([[0.0394]])
    kp_coord_xyz_root = torch.tensor([[ 0.0049, -0.0572,  0.7018]])
    keypoint_uv21_gt =  torch.tensor(
        [
            [
                [162.0000, 136.9000],
                [166.8000, 174.7000],
                [169.6000, 166.3000],
                [172.6000, 154.0000],
                # [168.6000, 145.4000],
                # [149.4000, 174.9000],
                # [152.6000, 167.9000],
                # [156.2000, 158.3000],
                # [160.8000, 150.5000],
                # [148.2000, 191.3000],
                # [147.3000, 177.4000],
                # [146.9000, 164.3000],
                # [150.4000, 151.9000],
                # [142.7000, 188.6000],
                # [141.8000, 174.8000],
                # [141.4000, 163.7000],
                # [144.2000, 152.9000],
                # [131.6000, 182.6000],
                # [131.8000, 175.2000],
                # [132.5000, 167.2000],
                # [138.6000, 155.1000]
            ]
        ])
    # keypoint_uv21 = forward_kinematics.project(keypoint_xyz21_normed, camera_intrinsic_matrix, keypoint_scale, kp_coord_xyz_root)
    keypoint_xyz21_absolute = forward_kinematics.convert_rel_normalized_to_absolute(keypoint_xyz21_rel_normed, keypoint_scale, kp_coord_xyz_root)
    keypoint_uv21 = forward_kinematics.project_xyz_to_uv(keypoint_xyz21_absolute, camera_intrinsic_matrix)
    print('keypoint_uv21.shape', keypoint_uv21.shape)
    # print(f'positions_uv[0,0:3]: {positions_uv[0,0:3]}')
    print('keypoint_uv21\n', keypoint_uv21)



    root_angles = torch.tensor([[0,0,math.pi/2]])
    other_angles = torch.tensor([[0,math.pi/2,0,     0,0,0,      0,
                                  math.pi/2,0,     0,      0,
                                  0,0,     0,      0,
                                  0,0,     0,      0,
                                  0,0,     0,      0,                                  
                                  ]])

    other_angles = torch.tensor([[0,math.pi/2,0,     0,0,0,      0,
                                  0,0,     0,      0,
                                  0,0,     0,      0,
                                  0,0,     0,      0,
                                  0,0,     0,      0,                                  
                                  ]])
    
    bone_lengths = torch.tensor([
        [1,1,1,1,
         1,1,1,1,
         1,1,1,1,
         1,1,1,1,
         1,1,1,1]])
    camera_intrinsic_matrix = torch.tensor(
    [[[320.9000,   0.0000, 160.0000],
         [  0.0000, 320.9000, 160.0000],
         [  0.0000,   0.0000,   1.0000]]], dtype=torch.float32)

    index_root_bone_length = torch.tensor([[1]])
    kp_coord_xyz_root = torch.tensor([[1,1,1]])
    kp_coord_xyz21_absolute, kp_coord_uv21 = forward_kinematics.forward(root_angles, other_angles, bone_lengths, camera_intrinsic_matrix, index_root_bone_length, kp_coord_xyz_root)
    print('kp_coord_xyz21_absolute', kp_coord_xyz21_absolute)
    print('kp_coord_uv21', kp_coord_uv21)
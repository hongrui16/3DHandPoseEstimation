import torch
import math
from config.config import *

'''
Thumb: there are 3 DOF at each of thumb carpometacarpal(CMC) and metacarpophalangeal (MCP) joints.The thumb interphalangeal (IP) joint also has
1 DOF, 7 DOF in total.

Fingers except thumb: For each such finger, there are two key points from the fingertip to the palm 
(excluding the connection with the palm). Each key point can do flexion/extension, which constitutes 2 DOF . 
In addition, the place where each finger is connected to the palm can be used for both flexion/extension and abduction/adduction, 
which constitutes 2 DOF. Added up, the total is 4 DOF. Since we have 4 of these Fingers, a total of 4 * 4 = 16 DOF.

right hand coordinate system
X-axis: Points in the direction of your thumb, horizontally to the right.
Y-axis: perpendicular to the direction of the palm, upward.
Z-axis: Points in the direction of the tip of the middle finger, forward.

If the middle finger is bent 10 degrees downward, it is a rotation around the X-axis
The expansion/merging of the index finger (Abduction/Adduction) is performed around the Y-axis.
If the entire palm rotates about the direction of the tip of the middle finger (i.e. the Z-axis), then this is a rotation about the Z-axis.

nodes = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4']

        wrist(root)
            |
            v
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

'''


class BoneAnglePrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = keypoint_num*3):
        super(BoneAnglePrediction, self).__init__()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, 3),  # Predicting root joint orientation
            torch.nn.Sigmoid()
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, other_joint_angles_num),  # Predicting other joint angles 
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        root_angles = self.mlp1(x)
        # Scale root_angles to the range of [-π, π]
        root_angles = root_angles * 2 * math.pi - math.pi

        other_angles = self.mlp2(x)
        # Scale other_angles to the range of [0, π/2]
        other_angles = other_angles * math.pi - math.pi/2


        return root_angles, other_angles
    

class BoneLengthPrediction(torch.nn.Module):
    def __init__(self, device = 'cpu', input_dim = keypoint_num*3):
        super(BoneLengthPrediction, self).__init__()
        self.device = device

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim, bone_length_num),  # Predicting length
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # bone_length = self.mlp1(x) * 1.5
        # # Create a tensor of ones with the same batch size as bone_length and unsqueeze to add a dimension
        # ones = torch.ones(bone_length.size(0), 1, device=self.device)
        # # Concatenate the ones tensor to the bone_length tensor
        # bone_length = torch.cat((ones, bone_length), dim=1)
        bone_length = self.mlp1(x)
        return bone_length



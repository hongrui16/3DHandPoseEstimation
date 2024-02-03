import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('../')  


from network.sub_modules.conditionalDiffusion import *
from config.config import *



class DiffusionJointEstimation(torch.nn.Module):
    def __init__(self):
        super(DiffusionJointEstimation, self).__init__()
        # You will need to define the input channels and output channels based on your specific problem.
        # self.unet = UNet()
        self.Unet1D_Model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1,
        condition_feat_dim = condition_feat_dim
        )

        self.diffusionModel = GaussianDiffusion1D(
            self.Unet1D_Model,
            seq_length = keypoint_num*3,
            timesteps = num_timesteps,
            sampling_timesteps= num_sampling_timesteps,
        )
        self.batch_size = batch_size
    
    def diffusion_loss(self, x, resnet_features):
        # Concatenate image features, 3D points, and time-step information
        # You may need to adjust the dimensions for concatenation based on your specific inputs.
        diffusion_loss = self.diffusionModel(x, resnet_features)
        return diffusion_loss

    def joint_coord_sampling(self, resnet_features):
        bs = resnet_features.shape[0]
        joint_coord = self.diffusionModel.sample(batch_size=bs, condition=resnet_features)
        return joint_coord


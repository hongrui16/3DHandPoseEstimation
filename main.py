import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import glob

from config.config import *

# from network.sub_modules.conditionalDiffusion import *
# from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
# from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
# from network.sub_modules.forwardKinematicsLayer import ForwardKinematics

from network.diffusion3DHandPoseEstimation import Diffusion3DHandPoseEstimation
from dataloader.dataloaderRHD import RHD_HandKeypointsDataset
from criterions.loss import LossCalculation


class Worker(object):
    def __init__(self):
                
        flag = torch.cuda.is_available()
        if flag:
            print("CUDA is available")
            device = "cuda"
        else:
            print("CUDA is unavailable")
            device = "cpu"
        self.device = device
        
        self.model = Diffusion3DHandPoseEstimation(device=device)

        self.criterion = LossCalculation(device=device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


        if dataset_name == 'RHD':
            train_set = RHD_HandKeypointsDataset(root_dir=dataset_root_dir, set_type='training')
            val_set = RHD_HandKeypointsDataset(root_dir=dataset_root_dir, set_type='evaluation')

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        log_dir = sorted(glob.glob(os.path.join(save_log_dir, dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        self.exp_dir = os.path.join(save_log_dir, dataset_name, f'run_{run_id:03d}')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.logger = SummaryWriter(self.exp_dir)

        self.min_val_epoch_loss = float('inf')
        self.training_epoch_loss = None
        self.val_epoch_loss = None
        
    def trainval(self, cur_epoch, total_epoch, model, loader, criterion, optimizer, split):
        assert split in ['training', 'validation']
        if split == 'training':
            model.train()
        else:
            model.eval()
        tbar = tqdm(loader)
        num_iter = len(loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)
        epoch_loss = []

        for iter, sample in enumerate(tbar):

            keypoint_uv21 = sample['keypoint_uv21'].to(self.device)
            keypoint_xyz21_normed = sample['keypoint_xyz21_normed'].to(self.device)
            keypoint_vis21 = sample['keypoint_vis21'].to(self.device)
            image = sample['image'].to(self.device)
            camera_matrix = sample['cam_mat'].to(self.device)
            index_root_bone_length = sample['keypoint_scale'].to(self.device)

            optimizer.zero_grad()

            refined_joint_coord, diffusion_loss, resnet_features = model(image, camera_matrix, keypoint_xyz21_normed)
            pred_xyz_coordinates, pred_uv_coordinates = refined_joint_coord

            loss_part1 = criterion(pred_xyz_coordinates, keypoint_xyz21_normed, pred_uv_coordinates, keypoint_uv21, keypoint_vis21) 
            loss = loss_part1 + diffusion_loss
            if split == 'training':
                loss.backward()
                optimizer.step()

            if iter % 10 == 0:
                tbar.set_description(f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Epoch: {iter:05d}/{num_iter:05d} loss: {loss:.5f}')
                self.logger.add_scalar(f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Epoch: {iter:05d}/{num_iter:05d} loss: {loss:.5f}')


            iter_loss = round(loss.item(), 5)
            epoch_loss.append(iter_loss)

        return np.array(epoch_loss).mean()
        
    
    def save_checkpoint(self, state, is_best, model_name='', ouput_weight_dir = ''):
        """Saves checkpoint to disk"""
        os.makedirs(ouput_weight_dir, exist_ok=True)
        best_model_filepath = os.path.join(ouput_weight_dir, f'{model_name}_model_best.pth.tar')
        filename = os.path.join(ouput_weight_dir, f'{model_name}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)
    
    def forward(self):
        for epoch in range(0, max_epoch): 
            self.training_epoch_loss = self.trainval(epoch, max_epoch, self.model, self.train_loader, self.criterion, self.optimizer, 'training')

            self.val_epoch_loss = self.trainval(epoch, max_epoch, self.model, self.val_loader, self.criterion, self.optimizer, 'validation')
            checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'val_epoch_loss': self.val_epoch_loss,                
                        }
            if self.val_epoch_loss < self.min_val_epoch_loss:
                self.min_val_epoch_loss = self.val_epoch_loss
                is_best = True
            else:
                is_best = False

            self.save_checkpoint(checkpoint, is_best, 'DF', self.exp_dir)


if __name__ == '__main__':
    worker = Worker()
    worker.forward()
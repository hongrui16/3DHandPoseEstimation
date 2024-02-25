import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import glob
import shutil
import GPUtil
import time
from datetime import datetime

from config import config

# from network.sub_modules.conditionalDiffusion import *
# from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
# from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
# from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
# from network.diffusion3DHandPoseEstimation import Diffusion3DHandPoseEstimation
# from network.twoDimHandPoseEstimation import TwoDimHandPoseEstimation, TwoDimHandPoseWithFKEstimation
# from network.threeDimHandPoseEstimation import ThreeDimHandPoseEstimation, OnlyThreeDimHandPoseEstimation
# from network.MANO3DHandPoseEstimation import MANO3DHandPoseEstimation
from network.hand3DPoseNet import Hand3DPoseNet
from network.hand3DPosePriorNetwork import Hand3DPosePriorNetwork

from dataloader.RHD.dataloaderRHD import RHD_HandKeypointsDataset
from criterions.loss import LossCalculation
from criterions.metrics import MPJPE
from utils.get_gpu_info import *

config.is_inference = False

class Worker(object):
    def __init__(self, gpu_index = None):
        
        cuda_valid = torch.cuda.is_available()
        if cuda_valid:
            gpu_index = gpu_index  # # Here set the index of the GPU you want to use
            print(f"CUDA is available, using GPU {gpu_index}")
            if config.gpu_idx is None:
                device = torch.device(f"cuda")
            else:
                device = torch.device(f"cuda:{gpu_index}")
        else:
            print("CUDA is unavailable, using CPU")
            device = torch.device("cpu")
        
        assert config.model_name in ['Hand3DPoseNet', 'Hand3DPosePriorNetwork']

        self.device = device

        if config.model_name == 'Hand3DPoseNet':
            self.model = Hand3DPoseNet(device)
            comp_xyz_loss = True
        elif config.model_name == 'Hand3DPosePriorNetwork':
            self.model = Hand3DPosePriorNetwork()
            comp_xyz_loss = True
            
        
        

            
        self.criterion = LossCalculation(device=device, comp_xyz_loss = comp_xyz_loss)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.metric_mpjpe = MPJPE()

        if config.dataset_name == 'RHD':
            train_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='training')
            val_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='evaluation')
        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=15)
        self.val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=15)
        
        current_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


        # log_dir_last_name = sorted(glob.glob(os.path.join(save_log_dir, dataset_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
        # run_id = int(log_dir_last_name[-1].split('_')[-1]) + 1 if log_dir_last_name else 0

        self.exp_dir = os.path.join(config.save_log_dir, config.model_name, config.dataset_name, f'run_{current_timestamp}')
        os.makedirs(self.exp_dir, exist_ok=True)

        self.txtfile = os.path.join(self.exp_dir, 'log.txt')
        if os.path.exists(self.txtfile):
            os.remove(self.txtfile)

        self.write_loginfo_to_txt(f'{self.exp_dir}')

        self.logger = SummaryWriter(self.exp_dir)

        self.best_val_epoch_mpjpe = float('inf')
        self.start_epoch = 0
        
        if config.resume_weight_path is not None:
            if not os.path.isfile(config.resume_weight_path):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(config.resume_weight_path))
            checkpoint = torch.load(config.resume_weight_path, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint["state_dict"])
            # Using the following load method will cause each process to occupy an extra part of the video memory on GPU0. The reason is that the default load location is GPU0.
            # checkpoint = torch.load("checkpoint.pth")

            # Update the model's state dict
            new_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in self.model.state_dict()}
            self.model.load_state_dict(new_state_dict, strict=False)

           
            # if cuda_valid:
            #     self.model.module.load_state_dict(checkpoint['state_dict'])
            # else:
            #     self.model.load_state_dict(checkpoint['state_dict'])
            
            # Check if the models are different
            old_keys = set(checkpoint['state_dict'].keys())
            new_keys = set(self.model.state_dict().keys())

            # If there's a difference in the keys, we assume the architectures are different
            if old_keys != new_keys:
                finetune = True
            else:
                finetune = False  # or set finetune based on some other condition or user input

            # Conditional loading of the optimizer state
            if not finetune: #train                
                self.best_val_epoch_mpjpe = checkpoint['MPJPE']
                self.start_epoch = checkpoint['epoch']

                # However, if you do want to load the state dict, you would need to ensure that the state matches the new model
                optimizer_state_dict = checkpoint['optimizer']
                
                # Filter out optimizer state that doesn't match the new model's parameters
                filtered_optimizer_state_dict = {
                    k: v for k, v in optimizer_state_dict.items() if k in self.optimizer.state_dict()
                }
                
                # Load the filtered state dict
                self.optimizer.load_state_dict(filtered_optimizer_state_dict)


            print("=> loaded checkpoint '{}' (epoch {})".format(config.resume_weight_path, checkpoint['epoch']))
            self.write_loginfo_to_txt("=> loaded checkpoint '{}' (epoch {})".format(config.resume_weight_path, checkpoint['epoch'])+'\n\n')
            # Clear start epoch if fine-tuning
            if finetune:
                self.start_epoch = 0

        self.model.to(device)
        shutil.copy('config/config.py', f'{self.exp_dir}/config.py')

        
    def trainval(self, cur_epoch, total_epoch, loader, split, fast_debug = False):
        assert split in ['training', 'validation']
        if split == 'training':
            self.model.train()
        else:
            self.model.eval()
        tbar = tqdm(loader)
        num_iter = len(loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)
        epoch_loss = []
        epoch_mpjpe = []



        for idx, sample in enumerate(tbar): # 6 ~ 10 s
            if fast_debug and iter > 2:
                break
            # if idx < 112:
            #     continue
            # print('idx', idx)
            if config.hand_crop:
                image = sample['image_crop'].to(self.device)
            else:
                image = sample['image'].to(self.device)

            keypoint_vis21_gt = sample['keypoint_vis21'].to(self.device) # visiable points mask
            kp_coord_xyz21_rel_can_gt = sample['kp_coord_xyz21_rel_can'].to(self.device) #scale length
            rot_mat_gt = sample['rot_mat'].to(self.device) #scale length
            scoremap = sample['scoremap'].to(self.device) #scale length
            # keypoint_xyz_root = sample['keypoint_xyz_root'].to(self.device)
            # keypoint_uv21_gt = sample['keypoint_uv21'].to(self.device) # uv coordinate
            # keypoint_xyz21_gt = sample['keypoint_xyz21'].to(self.device) # xyz absolute coordinate
            # keypoint_xyz21_rel_normed_gt = sample['keypoint_xyz21_rel_normed'].to(self.device) ## normalized xyz coordinates

            # camera_intrinsic_matrix = sample['camera_intrinsic_matrix'].to(self.device)
            
        
            self.optimizer.zero_grad()
            if split == 'training':
                if config.model_name == 'Hand3DPoseNet':
                    input = image
                elif config.model_name == 'Hand3DPosePriorNetwork':
                    input = scoremap
                result, _ = self.model(input)
                coord_xyz_rel_normed, can_xyz_kps21_pred, rot_mat_pred = result
                mpjpe = None
            else:
                with torch.no_grad():
                    result, _ = self.model(input)
                    coord_xyz_rel_normed, can_xyz_kps21_pred, rot_mat_pred = result

                    mpjpe = self.metric_mpjpe(can_xyz_kps21_pred, kp_coord_xyz21_rel_can_gt, keypoint_vis21_gt)
            
            # print('keypoint_xyz21_gt[0]', keypoint_xyz21_gt[0])
            # print('keypoint_xyz21_pred[0]', keypoint_xyz21_pred[0])
            
            # print('keypoint_uv21_gt[0]', keypoint_uv21_gt[0])
            # print('keypoint_uv21_pred[0]', keypoint_uv21_pred[0])
            # print('keypoint_uv21_pred.shape', keypoint_uv21_pred.shape)

            loss_xyz, _, _ = self.criterion(can_xyz_kps21_pred, kp_coord_xyz21_rel_can_gt, None, None, keypoint_vis21_gt) 
            loss_rot = torch.mean(torch.square(rot_mat_pred - rot_mat_gt))
            # print('loss_xyz', loss_xyz)
            # print('loss_rot', loss_rot)
            loss = loss_xyz + loss_rot
            if split == 'training':
                loss.backward()
                self.optimizer.step()

            if split == 'training':
                # loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}'
                loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}| L_xyz: {loss_xyz.item():.4f}, L_rot: {loss_rot.item():.4f}'
                tbar.set_description(loginfo)
            else:
                # loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f} MPJPE: {mpjpe.item():.4f}'
                loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f} MPJPE: {mpjpe.item():.4f}| L_xyz: {loss_xyz.item():.4f}, L_rot: {loss_rot.item():.4f}'
                tbar.set_description(loginfo)

            # if idx % 20 == 0:
            #     self.write_loginfo_to_txt(loginfo)
            # if iter % 50 == 0:
            #     gpu_info = get_gpu_utilization_as_string()
            #     print(gpu_info)
            #     self.write_loginfo_to_txt(gpu_info)
            


            iter_loss_value = round(loss.item(), 5)
            epoch_loss.append(iter_loss_value)
            if not split == 'training':
                iter_mpjpe_value = round(mpjpe.item(), 5)
                epoch_mpjpe.append(iter_mpjpe_value)
                    
        if not split == 'training':
            self.logger.add_scalar(f'{formatted_split} epoch MPJPE', np.round(np.mean(epoch_mpjpe), 5), global_step=cur_epoch)
            epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {np.round(np.mean(epoch_loss), 4)}, MPJPE: {np.round(np.mean(epoch_mpjpe), 5)}'            
            epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        else:
            self.logger.add_scalar(f'{formatted_split} epoch loss', np.round(np.mean(epoch_loss), 5), global_step=cur_epoch)
            epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {np.round(np.mean(epoch_loss), 4)}'
            epoch_mpjpe = None
        print(epoch_info)
        self.write_loginfo_to_txt(epoch_info)
        self.write_loginfo_to_txt('')
        return epoch_mpjpe
    
    def save_checkpoint(self, state, is_best, model_name='', ouput_weight_dir = ''):
        """Saves checkpoint to disk"""
        os.makedirs(ouput_weight_dir, exist_ok=True)
        best_model_filepath = os.path.join(ouput_weight_dir, f'{model_name}_model_best.pth.tar')
        filename = os.path.join(ouput_weight_dir, f'{model_name}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:   
            torch.save(state, best_model_filepath)
    
    def write_loginfo_to_txt(self, loginfo):
        loss_file = open(self.txtfile, "a+")
        if loginfo.endswith('\n'):
            loss_file.write(loginfo)
        else:
            loss_file.write(loginfo+'\n')
        loss_file.close()# 
    
    def forward(self, fast_debug = False):
        for epoch in range(self.start_epoch, config.max_epoch): 
            # _ = self.trainval(epoch, max_epoch, self.val_loader, 'training', fast_debug = fast_debug)
            _ = self.trainval(epoch, config.max_epoch, self.train_loader, 'training', fast_debug = fast_debug)

            mpjpe = self.trainval(epoch, config.max_epoch, self.val_loader, 'validation', fast_debug = fast_debug)
            checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'MPJPE': mpjpe,                
                        }
            if mpjpe < self.best_val_epoch_mpjpe:
                self.best_val_epoch_mpjpe = mpjpe
                is_best = True
            else:
                is_best = False

            self.save_checkpoint(checkpoint, is_best, 'DF', self.exp_dir)
            print('')


if __name__ == '__main__':
    # fast_debug = True
    fast_debug = False
    worker = Worker(config.gpu_idx)
    worker.forward(fast_debug)

    # gpu_info = get_gpu_utilization_as_string()
    # print('gpu_info', gpu_info)

# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00

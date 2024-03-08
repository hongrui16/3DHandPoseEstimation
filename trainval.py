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
import platform
import argparse

from config import config

# from network.sub_modules.conditionalDiffusion import *
# from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
# from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
# from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.DiffusionHandPose import DiffusionHandPose
from network.MANO3DHandPose import MANO3DHandPose
from network.OnlyThreeDimHandPose import OnlyThreeDimHandPose
from network.Resnet50MANO3DHandPose import Resnet50MANO3DHandPose
from network.ThreeDimHandPose import ThreeDimHandPose
from network.ThreeHandShapeAndPoseMANO import ThreeHandShapeAndPoseMANO
from network.TwoDimHandPose import TwoDimHandPose
from network.TwoDimHandPoseWithFK import TwoDimHandPoseWithFK



from dataloader.RHD.dataloaderRHD import RHD_HandKeypointsDataset
from criterions.loss import LossCalculation
from criterions.metrics import MPJPE
from utils.get_gpu_info import *
from utils.coordinate_trans import batch_project_xyz_to_uv

config.is_inference = False

# if platform.system() == 'Windows':
#     print("This is Windows")
# elif platform.system() == 'Linux':
#     print("This is Linux")
# elif platform.system() == 'Darwin':
#     print("This is MacOS")

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
        
        # assert config.model_name in ['DiffusionHandPose', 'TwoDimHandPose', 'ThreeDimHandPose', 'OnlyThreeDimHandPose', 
        #                              'TwoDimHandPoseWithFK', 'MANO3DHandPose', 'threeHandShapeAndPoseMANO']

        self.device = device
        self.comp_hand_mask_loss = False
        self.comp_regularization_loss = False
        self.comp_xyz_loss = False
        self.comp_uv_loss = False
        self.comp_diffusion_loss = False
        self.comp_contrast_loss = False
        
        if config.model_name == 'TwoDimHandPose':
            self.model = TwoDimHandPose(device)
            self.comp_xyz_loss = False
            self.comp_uv_loss = True
        elif config.model_name == 'TwoDimHandPoseWithFK':
            self.model = TwoDimHandPoseWithFK(device)
            self.comp_xyz_loss = True
            self.comp_uv_loss = True
        elif config.model_name == 'DiffusionHandPose':
            self.model = DiffusionHandPose(device)
            self.comp_xyz_loss = True
            self.comp_diffusion_loss = True
            self.comp_uv_loss = True
        elif config.model_name == 'ThreeDimHandPose':
            self.model = ThreeDimHandPose(device)
            self.comp_xyz_loss = True
            self.comp_uv_loss = True
        elif config.model_name == 'OnlyThreeDimHandPose':
            self.model = OnlyThreeDimHandPose(device)
            self.comp_xyz_loss = True 
        elif config.model_name == 'MANO3DHandPose':
            self.model = MANO3DHandPose(device)
            self.comp_xyz_loss = True
            self.comp_uv_loss = True
        elif config.model_name == 'ThreeHandShapeAndPoseMANO':
            self.model = ThreeHandShapeAndPoseMANO(device)
            self.comp_xyz_loss = True
            config.compute_uv_loss = False
            self.comp_uv_loss = False
        elif config.model_name == 'Resnet50MANO3DHandPose':
            self.model = Resnet50MANO3DHandPose(device)
            self.comp_xyz_loss = True
            self.comp_uv_loss = True
            self.comp_hand_mask_loss = True
            self.comp_regularization_loss = True
        else:
            raise ValueError(f'config.model_name {config.model_name} is not supported')
        
            
        self.criterion = LossCalculation(device=device, comp_xyz_loss = self.comp_xyz_loss, comp_uv_loss = self.comp_uv_loss, comp_hand_mask_loss = self.comp_hand_mask_loss, comp_regularization_loss = self.comp_regularization_loss)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.metric_mpjpe = MPJPE()

        if config.dataset_name == 'RHD':
            if platform.system() == 'Windows':
                train_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='evaluation')
                bs = 2
            elif platform.system() == 'Linux':
                if config.use_val_dataset_to_debug:
                    train_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='evaluation')
                    shuffle = False
                else:
                    train_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='training')
                    shuffle = True
                bs = config.batch_size
            val_set = RHD_HandKeypointsDataset(root_dir=config.dataset_root_dir, set_type='evaluation')
        self.train_loader = DataLoader(train_set, batch_size=bs, shuffle=shuffle, num_workers=config.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=config.num_workers)
        
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
        batch_size = 4
        self.input_shape = (batch_size, config.input_channels, 256, 256)
        self.kp_vis21_shape = (batch_size, 21, 1)
        self.kp_coord_xyz21_shape = (batch_size, 21, 3)
        self.kp_coord_21_shape = (batch_size, 21, 2)
        self.scoremap_shape = (batch_size, 21, 256, 256)
        self.hand_shape = (batch_size, 256, 256)
        self.camera_intrinsic_matrix_shape = (batch_size, 3, 3)
        self.kp_xyz_root_shape = (batch_size, 3)
        self.kp_scale_shape = (batch_size, 1)
        print(f'log dir: {self.exp_dir}')
        shutil.copy('config/config.py', f'{self.exp_dir}/config.py')

        
    def trainval_real(self, cur_epoch, total_epoch, loader, split, fast_debug = False):
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
        epoch_loss_diff = []
        epoch_loss_xyz = []
        epoch_loss_uv = []
        epoch_loss_contrast = []
        epoch_loss_hand_mask = []
        epoch_loss_regularization = []
        epoch_mpjpe = []

        # data_iter = iter(tbar)  # 创建 DataLoader 的迭代器
        # for idx in tqdm(range(len(tbar))):
        #     start_time = time.time()  # 开始时间

        #     # 使用 next 从 DataLoader 获取下一个 batch
        #     try:
        #         sample = next(data_iter)
        #     except StopIteration:
        #         break  # 如果 DataLoader 结束，则退出循环

        #     end_time = time.time()  # 结束时间
        #     elapsed_time = end_time - start_time  # 计算所用时间
        #     print(f"Iteration {idx} took {elapsed_time} seconds to retrieve one batch.") # 6 ~ 10 s



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
            index_root_bone_length = sample['keypoint_scale'].to(self.device) #scale length
            keypoint_xyz_root = sample['keypoint_xyz_root'].to(self.device)
            keypoint_uv21_gt = sample['keypoint_uv21'].to(self.device) # uv coordinate
            keypoint_xyz21_gt = sample['keypoint_xyz21'].to(self.device) # xyz absolute coordinate
            keypoint_xyz21_rel_normed_gt = sample['keypoint_xyz21_rel_normed'].to(self.device) ## normalized xyz coordinates
            scoremap = sample['scoremap'].to(self.device) #scale length

            camera_intrinsic_matrix = sample['camera_intrinsic_matrix'].to(self.device)
            gt_hand_mask = sample['right_hand_mask'].to(self.device)
            
            if config.model_name == 'Resnet50MANO3DHandPose':
                input = torch.cat([image, scoremap], dim=1)
            else:
                input = image
            bs, num_points, c = keypoint_xyz21_rel_normed_gt.shape
            # print('keypoint_xyz21_rel_normed_gt.shape', keypoint_xyz21_rel_normed_gt.shape)
            pose_x0 = keypoint_xyz21_rel_normed_gt.view(bs, -1, num_points*c)
            # print('pose_x0.shape', pose_x0.shape)
            # print('index_root_bone_length.shape', index_root_bone_length.shape)

            self.optimizer.zero_grad()
            if split == 'training':
                refined_joint_coord, loss_diffusion, theta_beta = self.model(input, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root, pose_x0)
                keypoint_xyz21_pred, keypoint_uv21_pred, _ = refined_joint_coord
                mpjpe = None
            else:
                with torch.no_grad():
                    refined_joint_coord, loss_diffusion, theta_beta = self.model(input, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root, pose_x0)
                    keypoint_xyz21_pred, keypoint_uv21_pred, _ = refined_joint_coord
                    if config.model_name == 'TwoDimHandPose':
                        mpjpe = self.metric_mpjpe(keypoint_uv21_pred, keypoint_uv21_gt, keypoint_vis21_gt)
                    else:
                        # elif model_name == 'DiffusionHandPose' or model_name == 'ThreeDimHandPose':
                        mpjpe = self.metric_mpjpe(keypoint_xyz21_pred, keypoint_xyz21_gt, keypoint_vis21_gt)
            
            # print('keypoint_xyz21_gt[0]', keypoint_xyz21_gt[0])
            # print('keypoint_xyz21_pred[0]', keypoint_xyz21_pred[0])
            
            # print('keypoint_uv21_gt[0]', keypoint_uv21_gt[0])
            # print('keypoint_uv21_pred[0]', keypoint_uv21_pred[0])
            # print('keypoint_uv21_pred.shape', keypoint_uv21_pred.shape)
            theta, beta = theta_beta
            loss = torch.tensor(0, dtype=torch.float, device=self.device)
            loss_xyz, loss_uv, loss_contrast, loss_hand_mask, loss_regularization = self.criterion(keypoint_xyz21_pred, keypoint_xyz21_gt, keypoint_uv21_pred, keypoint_uv21_gt, keypoint_vis21_gt, hand_mask = gt_hand_mask, theta = theta, beta = beta) 
            
            loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}'
            if not split == 'training':                            
                loginfo += f'| MPJPE: {mpjpe.item():.4f}'
            if self.comp_diffusion_loss:
                loginfo += f'| L_diff: {loss_diffusion.item():.4f}'                
                epoch_loss_diff.append(loss_diffusion.item())
                loss += loss_diffusion
            if self.comp_xyz_loss:
                loginfo += f'| L_xyz: {loss_xyz.item():.4f}'
                epoch_loss_xyz.append(loss_xyz.item())
                loss += loss_xyz
            if self.comp_uv_loss:
                loginfo += f'| L_uv: {loss_uv.item():.4f}'
                epoch_loss_uv.append(loss_uv.item())
                loss += loss_uv/100000
            if self.comp_contrast_loss:
                loginfo += f'| L_cont: {loss_contrast.item():.4f}'
                epoch_loss_contrast.append(loss_contrast.item())
                loss += loss_contrast
            if self.comp_hand_mask_loss:    
                loginfo += f'| L_hmask: {loss_hand_mask.item():.4f}'
                epoch_loss_hand_mask.append(loss_hand_mask.item())
                loss += loss_hand_mask
            if self.comp_regularization_loss:
                loginfo += f'| L_regu: {loss_regularization.item():.4f}'
                epoch_loss_regularization.append(loss_regularization.item())
                loss += loss_regularization
        
            if split == 'training':
                loss.backward()
                self.optimizer.step()

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
            
            del loss
            # if config.use_val_dataset_to_debug:
            #     break
        epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {np.round(np.mean(epoch_loss), 4)}'     
        if self.comp_diffusion_loss:
            epoch_info += f'| L_diff: {np.round(np.mean(epoch_loss_diff), 4)}'
        if self.comp_xyz_loss:
            epoch_info += f'| L_xyz: {np.round(np.mean(epoch_loss_xyz), 4)}'
        if self.comp_uv_loss:
            epoch_info += f'| L_uv: {np.round(np.mean(epoch_loss_uv), 4)}'
        if self.comp_contrast_loss:
            epoch_info += f'| L_cont: {np.round(np.mean(epoch_loss_contrast), 4)}'
        if self.comp_hand_mask_loss:    
            epoch_info += f'| L_hmask: {np.round(np.mean(epoch_loss_hand_mask), 4)}'
        if self.comp_regularization_loss:
            epoch_info += f'| L_regu: {np.round(np.mean(epoch_loss_regularization), 4)}'
        if not split == 'training':
            self.logger.add_scalar(f'{formatted_split} epoch MPJPE', np.round(np.mean(epoch_mpjpe), 5), global_step=cur_epoch)
            epoch_info +=  f'\nMPJPE: {np.round(np.mean(epoch_mpjpe), 5)}' 
            epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        else:
            self.logger.add_scalar(f'{formatted_split} epoch loss', np.round(np.mean(epoch_loss), 5), global_step=cur_epoch)
            epoch_mpjpe = None

        print(epoch_info)
        self.write_loginfo_to_txt(epoch_info)
        return epoch_mpjpe

    
        
    def trainval_fake(self, cur_epoch, total_epoch, loader, split, fast_debug = False):
        assert split in ['training', 'validation']
        if split == 'training':
            self.model.train()
        else:
            self.model.eval()

        num_iter = 20
        tbar = tqdm(range(num_iter))

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)
        epoch_loss = []
        epoch_loss_diff = []
        epoch_loss_xyz = []
        epoch_loss_uv = []
        epoch_loss_contrast = []
        epoch_loss_hand_mask = []
        epoch_loss_regularization = []
        epoch_mpjpe = []

        for idx in tbar: # 6 ~ 10 s
            if fast_debug and iter > 2:
                break
            
            image = torch.zeros(self.input_shape).to(self.device) + 0.5
            bs, c, h, w = image.shape
            image[:, :, -h//2:] = -0.5
            keypoint_vis21_gt = torch.ones(self.kp_vis21_shape, dtype=torch.bool, device=self.device)
            index_root_bone_length = torch.ones(self.kp_scale_shape, device=self.device)
            keypoint_xyz_root = torch.zeros(self.kp_xyz_root_shape).to(self.device)
            keypoint_xyz21_gt = torch.zeros(self.kp_coord_xyz21_shape).to(self.device) + 0.5
            keypoint_xyz21_gt[:, 0] = 0
            keypoint_xyz21_gt[:, -10:] = -0.5
            keypoint_xyz21_rel_normed_gt = keypoint_xyz21_gt
            scoremap = torch.zeros(self.scoremap_shape).to(self.device)
            camera_intrinsic_matrix = torch.zeros(self.camera_intrinsic_matrix_shape).to(self.device)
            camera_intrinsic_matrix[:, 0, 0] = 600
            camera_intrinsic_matrix[:, 1, 1] = 600
            camera_intrinsic_matrix[:, 0, 2] = 300
            camera_intrinsic_matrix[:, 1, 2] = 300
            keypoint_uv21_gt = batch_project_xyz_to_uv(keypoint_xyz21_gt, camera_intrinsic_matrix)
            gt_hand_mask = torch.ones(self.hand_shape, dtype=torch.bool, device=self.device)

            if config.model_name == 'Resnet50MANO3DHandPose':
                input = torch.cat([image, scoremap], dim=1)
            else:
                input = image
            bs, num_points, c = keypoint_xyz21_rel_normed_gt.shape
            # print('keypoint_xyz21_rel_normed_gt.shape', keypoint_xyz21_rel_normed_gt.shape)
            pose_x0 = keypoint_xyz21_rel_normed_gt.view(bs, -1, num_points*c)
            # print('pose_x0.shape', pose_x0.shape)
            # print('index_root_bone_length.shape', index_root_bone_length.shape)

            self.optimizer.zero_grad()
            if split == 'training':
                refined_joint_coord, loss_diffusion, theta_beta = self.model(input, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root, pose_x0)
                keypoint_xyz21_pred, keypoint_uv21_pred, _ = refined_joint_coord
                mpjpe = None
            else:
                with torch.no_grad():
                    refined_joint_coord, loss_diffusion, theta_beta = self.model(input, camera_intrinsic_matrix, index_root_bone_length, keypoint_xyz_root, pose_x0)
                    keypoint_xyz21_pred, keypoint_uv21_pred, _ = refined_joint_coord
                    if config.model_name == 'TwoDimHandPose':
                        mpjpe = self.metric_mpjpe(keypoint_uv21_pred, keypoint_uv21_gt, keypoint_vis21_gt)
                    else:
                        # elif model_name == 'DiffusionHandPose' or model_name == 'ThreeDimHandPose':
                        mpjpe = self.metric_mpjpe(keypoint_xyz21_pred, keypoint_xyz21_gt, keypoint_vis21_gt)
            
            # print('keypoint_xyz21_gt[0]', keypoint_xyz21_gt[0])
            # print('keypoint_xyz21_pred[0]', keypoint_xyz21_pred[0])
            
            # print('keypoint_uv21_gt[0]', keypoint_uv21_gt[0])
            # print('keypoint_uv21_pred[0]', keypoint_uv21_pred[0])
            # print('keypoint_uv21_pred.shape', keypoint_uv21_pred.shape)
            theta, beta = theta_beta
            loss = torch.tensor(0, dtype=torch.float, device=self.device)
            loss_xyz, loss_uv, loss_contrast, loss_hand_mask, loss_regularization = self.criterion(keypoint_xyz21_pred, keypoint_xyz21_gt, keypoint_uv21_pred, keypoint_uv21_gt, keypoint_vis21_gt, hand_mask = gt_hand_mask, theta = theta, beta = beta) 
            
            loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f}'
            if not split == 'training':                            
                loginfo += f'| MPJPE: {mpjpe.item():.4f}'
            if self.comp_diffusion_loss:
                loginfo += f'| L_diff: {loss_diffusion.item():.4f}'                
                epoch_loss_diff.append(loss_diffusion.item())
                loss += loss_diffusion
            if self.comp_xyz_loss:
                loginfo += f'| L_xyz: {loss_xyz.item():.4f}'
                epoch_loss_xyz.append(loss_xyz.item())
                loss += loss_xyz
            if self.comp_uv_loss:
                loginfo += f'| L_uv: {loss_uv.item():.4f}'
                epoch_loss_uv.append(loss_uv.item())
                loss += loss_uv/100000
            if self.comp_contrast_loss:
                loginfo += f'| L_cont: {loss_contrast.item():.4f}'
                epoch_loss_contrast.append(loss_contrast.item())
                loss += loss_contrast
            if self.comp_hand_mask_loss:    
                loginfo += f'| L_hmask: {loss_hand_mask.item():.4f}'
                epoch_loss_hand_mask.append(loss_hand_mask.item())
                loss += loss_hand_mask
            if self.comp_regularization_loss:
                loginfo += f'| L_regu: {loss_regularization.item():.4f}'
                epoch_loss_regularization.append(loss_regularization.item())
                loss += loss_regularization
        
            if split == 'training':
                loss.backward()
                self.optimizer.step()

            tbar.set_description(loginfo)



            iter_loss_value = round(loss.item(), 5)
            epoch_loss.append(iter_loss_value)
            if not split == 'training':
                iter_mpjpe_value = round(mpjpe.item(), 5)
                epoch_mpjpe.append(iter_mpjpe_value)
            
            del loss
            # if config.use_val_dataset_to_debug:
            #     break
        epoch_info = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Loss: {np.round(np.mean(epoch_loss), 4)}'     
        if self.comp_diffusion_loss:
            epoch_info += f'| L_diff: {np.round(np.mean(epoch_loss_diff), 4)}'
        if self.comp_xyz_loss:
            epoch_info += f'| L_xyz: {np.round(np.mean(epoch_loss_xyz), 4)}'
        if self.comp_uv_loss:
            epoch_info += f'| L_uv: {np.round(np.mean(epoch_loss_uv), 4)}'
        if self.comp_contrast_loss:
            epoch_info += f'| L_cont: {np.round(np.mean(epoch_loss_contrast), 4)}'
        if self.comp_hand_mask_loss:    
            epoch_info += f'| L_hmask: {np.round(np.mean(epoch_loss_hand_mask), 4)}'
        if self.comp_regularization_loss:
            epoch_info += f'| L_regu: {np.round(np.mean(epoch_loss_regularization), 4)}'
        if not split == 'training':
            self.logger.add_scalar(f'{formatted_split} epoch MPJPE', np.round(np.mean(epoch_mpjpe), 5), global_step=cur_epoch)
            epoch_info +=  f'\nMPJPE: {np.round(np.mean(epoch_mpjpe), 5)}' 
            epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        else:
            self.logger.add_scalar(f'{formatted_split} epoch loss', np.round(np.mean(epoch_loss), 5), global_step=cur_epoch)
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
            if config.use_fake_data:
                _ = self.trainval_fake(epoch, config.max_epoch, self.train_loader, 'training', fast_debug = fast_debug)
                mpjpe = self.trainval_fake(epoch, config.max_epoch, self.val_loader, 'validation', fast_debug = fast_debug)
                self.write_loginfo_to_txt('')
            else:
                _ = self.trainval_real(epoch, config.max_epoch, self.train_loader, 'training', fast_debug = fast_debug)
                mpjpe = self.trainval_real(epoch, config.max_epoch, self.val_loader, 'validation', fast_debug = fast_debug)
                self.write_loginfo_to_txt('')

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
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--gpuid', type=int, default=0, help='GPU index')
    parser.add_argument('--fast_debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    config.gpu_idx = args.gpuid
    fast_debug = args.fast_debug
    # fast_debug = True
    worker = Worker(config.gpu_idx)
    worker.forward(fast_debug)

    # gpu_info = get_gpu_utilization_as_string()
    # print('gpu_info', gpu_info)

# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
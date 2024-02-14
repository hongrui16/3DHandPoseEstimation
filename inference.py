        
import torch

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

from config.config import *

# from network.sub_modules.conditionalDiffusion import *
# from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
# from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
# from network.sub_modules.forwardKinematicsLayer import ForwardKinematics
from network.diffusion3DHandPoseEstimation import Diffusion3DHandPoseEstimation
from network.twoDimHandPoseEstimation import TwoDimHandPoseEstimation
from network.threeDimHandPoseEstimation import ThreeDimHandPoseEstimation, OnlyThreeDimHandPoseEstimation

from dataloader.RHD.dataloaderRHD import RHD_HandKeypointsDataset
from criterions.loss import LossCalculation
from criterions.metrics import MPJPE
from utils.get_gpu_info import *
from utils.plot_anno import *

class Worker(object):
    def __init__(self, gpu_index = None):
        cuda_valid = torch.cuda.is_available()
        if cuda_valid:
            gpu_index = gpu_index  # # Here set the index of the GPU you want to use
            print(f"CUDA is available, using GPU {gpu_index}")
            if gpu_idx is None:
                device = torch.device(f"cuda")
            else:
                device = torch.device(f"cuda:{gpu_index}")
        else:
            print("CUDA is unavailable, using CPU")
            device = torch.device("cpu")
        
        assert model_name in ['DiffusionHandPose', 'TwoDimHandPose', 'ThreeDimHandPose', 'OnlyThreeDimHandPose']

        self.device = device
        self.save_img = True

        if model_name == 'TwoDimHandPose':
            self.model = TwoDimHandPoseEstimation(device)
        elif model_name == 'DiffusionHandPose':
            self.model = Diffusion3DHandPoseEstimation(device)
        elif model_name == 'ThreeDimHandPose':
            self.model = ThreeDimHandPoseEstimation(device)
        elif model_name == 'OnlyThreeDimHandPose':
            self.model = OnlyThreeDimHandPoseEstimation(device)
            

        self.model.to(device)
            

        self.metric_mpjpe = MPJPE()

        if dataset_name == 'RHD':
            val_set = RHD_HandKeypointsDataset(root_dir=dataset_root_dir, set_type='evaluation')
        self.val_loader = DataLoader(val_set, batch_size=infer_batch_size, shuffle=False, num_workers=15)

        save_log_dir = resume_weight_path[:resume_weight_path.find(resume_weight_path.split('/')[-1])]
        log_dir = sorted(glob.glob(os.path.join(save_log_dir, 'infer_*')), key=lambda x: int(x.split('_')[-1]))
        run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
        self.exp_dir = os.path.join(save_log_dir, f'infer_{run_id:03d}')
        os.makedirs(self.exp_dir, exist_ok=True)

        self.img_save_dir = os.path.join(self.exp_dir, 'img')
        if self.save_img:
            os.makedirs(self.img_save_dir, exist_ok=True)

        self.txtfile = os.path.join(self.exp_dir, 'eval_log.txt')
        if os.path.exists(self.txtfile):
            os.remove(self.txtfile)

        self.logger = SummaryWriter(self.exp_dir)

        self.best_val_epoch_mpjpe = float('inf')
        self.start_epoch = 0
        
        if not os.path.isfile(resume_weight_path):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(resume_weight_path))
        checkpoint = torch.load(resume_weight_path, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint["state_dict"])
        # Using the following load method will cause each process to occupy an extra part of the video memory on GPU0. The reason is that the default load location is GPU0.
        # checkpoint = torch.load("checkpoint.pth")

        self.model.load_state_dict(checkpoint['state_dict'])
        
        # if cuda_valid:
        #     self.model.module.load_state_dict(checkpoint['state_dict'])
        # else:
        #     self.model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})".format(resume_weight_path, checkpoint['epoch']))


        shutil.copy('config/config.py', f'{self.exp_dir}/config.py')

        
    def eval(self, loader, split, fast_debug = False):
        assert split in ['validation']
        self.model.eval()
        tbar = tqdm(loader)
        num_iter = len(loader)

        width = 10  # Total width including the string length
        formatted_split = split.rjust(width)

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
            if hand_crop:
                images = sample['image_crop']
            else:
                images = sample['image']

            rgb_img = (255*(0.5+images[0].permute(1, 2, 0))).numpy().astype(np.uint8)

            images = images.to(self.device)
            keypoint_vis21_gt = sample['keypoint_vis21'].to(self.device) # visiable points mask
            index_root_bone_length = sample['keypoint_scale'].to(self.device) #scale length
            keypoint_xyz_root = sample['keypoint_xyz_root'].to(self.device)
            keypoint_uv21_gt = sample['keypoint_uv21'].to(self.device) # uv coordinate
            keypoint_xyz21_gt = sample['keypoint_xyz21'].to(self.device) # xyz absolute coordinate
            keypoint_xyz21_rel_normed_gt = sample['keypoint_xyz21_rel_normed'].to(self.device) ## normalized xyz coordinates
            camera_intrinsic_matrix = sample['camera_intrinsic_matrix'].to(self.device)
            img_names = sample['img_name']
            # print('img_names', img_names)
            # print('img_names[0]', img_names[0])
            
            
            bs, num_points, c = keypoint_xyz21_rel_normed_gt.shape
            # print('keypoint_xyz21_rel_normed_gt.shape', keypoint_xyz21_rel_normed_gt.shape)
            pose_x0 = keypoint_xyz21_rel_normed_gt.view(bs, -1, num_points*c)
            # print('pose_x0.shape', pose_x0.shape)
            # print('index_root_bone_length.shape', index_root_bone_length.shape)

        
            with torch.no_grad():
                refined_joint_coord, loss_diffusion = self.model(images, camera_intrinsic_matrix, pose_x0, index_root_bone_length, keypoint_xyz_root)
                keypoint_xyz21_pred, keypoint_uv21_pred = refined_joint_coord
                if model_name == 'TwoDimHandPose':
                    mpjpe = self.metric_mpjpe(keypoint_uv21_pred, keypoint_uv21_gt, keypoint_vis21_gt)
                else:
                    # elif model_name == 'DiffusionHandPose' or model_name == 'ThreeDimHandPose':
                    mpjpe = self.metric_mpjpe(keypoint_xyz21_pred, keypoint_xyz21_gt, keypoint_vis21_gt) 
                
            if self.save_img:
                img_filepath = os.path.join(self.img_save_dir, img_names[0].split('.')[0] + '_pre.jpg')
                plot_uv_on_image(keypoint_uv21_pred[0].cpu().numpy(), rgb_img, keypoints_vis = keypoint_vis21_gt[0].cpu().numpy().squeeze(), img_filepath = img_filepath)


            # loginfo = f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Iter: {idx:05d}/{num_iter:05d}, Loss: {loss.item():.4f} MPJPE: {mpjpe.item():.4f}'
            loginfo = f'{formatted_split} Iter: {idx:05d}/{num_iter:05d}, MPJPE: {mpjpe.item():.4f}'
            tbar.set_description(loginfo)

            # if idx % 20 == 0:
            #     self.write_loginfo_to_txt(loginfo)
            # if iter % 50 == 0:
            #     gpu_info = get_gpu_utilization_as_string()
            #     print(gpu_info)
            #     self.write_loginfo_to_txt(gpu_info)
            


        iter_mpjpe_value = round(mpjpe.item(), 5)
        epoch_mpjpe.append(iter_mpjpe_value)
                
        epoch_info = f'{formatted_split} MPJPE: {np.round(np.mean(epoch_mpjpe), 5)}'            
        epoch_mpjpe = np.round(np.mean(epoch_mpjpe), 5)
        print(epoch_info)
        self.write_loginfo_to_txt(epoch_info)
        return epoch_mpjpe
    
    
    def write_loginfo_to_txt(self, loginfo):
        loss_file = open(self.txtfile, "a+")
        if loginfo.endswith('\n'):
            loss_file.write(loginfo)
        else:
            loss_file.write(loginfo+'\n')
        loss_file.close()# 
    
    def forward(self, fast_debug = False):
        mpjpe = self.eval(self.val_loader, 'validation', fast_debug = fast_debug)


if __name__ == '__main__':
    # fast_debug = True
    fast_debug = False
    gpu_idx = 1
    gpu_idx = None
    worker = Worker(gpu_idx)
    worker.forward(fast_debug)

    # gpu_info = get_gpu_utilization_as_string()
    # print('gpu_info', gpu_info)

# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.80gb:1 --mem=80gb -t 0-24:00:00
# salloc -p gpuq -q gpu --nodes=1 --ntasks-per-node=30 --gres=gpu:A100.40gb:1 --mem=50gb -t 0-24:00:00
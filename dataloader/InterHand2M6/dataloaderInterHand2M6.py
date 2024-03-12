# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio
import sys, os
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
sys.path.append('../..')

from config import config
from config import config as cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints
from utils.compute_heatmap import render_gaussian_heatmap
from utils.plot_anno import *
from utils.general import crop_image_from_xy_torch
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.relative_trafo import bone_rel_trafo
from utils.coordinate_trans import camera_xyz_to_uv

class DataloaderInterHand2M6(Dataset):
    def __init__(self, transform, mode):
        assert mode in ('train', 'test', 'val')
        self.mode = mode # train, test, val
        self.dataset_root_dir = '/scratch/rhong5/dataset/InterHand/InterHand2.6M'
        self.img_path = f'{self.dataset_root_dir}/images'
        self.annot_path =  f'{self.dataset_root_dir}/annotations'
        if self.mode == 'val':
            self.rootnet_output_path =  f'{self.dataset_root_dir}/rootnet_output/rootnet_interhand2.6m_output_val.json'
        elif self.mode == 'test':
            self.rootnet_output_path =  f'{self.dataset_root_dir}/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.rootnet_output_dir =  f'{self.dataset_root_dir}/rootnet_output'
        os.makedirs(self.rootnet_output_dir, exist_ok=True)

        self.transform = transform
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")
        
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])
            
            campos = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32)
            camrot = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32)
            princpt = np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            
            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            cam_param = {'focal': focal, 
                         'princpt': princpt}
            
            joint = {'cam_coord': joint_cam, 
                     'img_coord': joint_img, 
                     'valid': joint_valid}
            
            data = {'img_path': img_path, 
                    'seq_name': seq_name, 
                    'cam_param': cam_param, 
                    'bbox': bbox, 
                    'joint': joint, 
                    'hand_type': hand_type, 
                    'hand_type_valid': hand_type_valid, 
                    'abs_depth': abs_depth, 
                    'file_name': img['file_name'], 
                    'capture': capture_id, 
                    'cam': cam, 
                    'frame': frame_idx}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))


        # general parameters
        self.sigma = config.sigma
        self.shuffle = config.shuffle
        self.use_wrist_coord = config.use_wrist_coord
        self.random_crop_to_size = config.random_crop_to_size
        self.random_crop_size = 256
        self.scale_to_size = config.scale_to_size
        self.scale_target_size = (240, 320)  # size its scaled down to if scale_to_size=True

        # data augmentation parameters
        self.hue_aug = config.hue_aug
        self.hue_aug_max = 0.1

        self.hand_crop = config.hand_crop
        self.coord_uv_noise = config.coord_uv_noise
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = config.crop_center_noise
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints

        self.crop_scale_noise = config.crop_scale_noise
        self.crop_offset_noise = config.crop_offset_noise
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)

        self.calculate_scoremap = config.calculate_scoremap
        self.scoremap_dropout = config.scoremap_dropout
        self.scoremap_dropout_prob = 0.8

        # these are constants of the dataset and therefore must not be changed
        self.image_size = (320, 320)
        self.crop_size = 256
        self.num_kp = 42


    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    

    def convert_joint_order_from_InterHand2M6_to_RHD(self, InterHand2M6_joint):
        # Mapping of the joint order of InterHand2M6 to the joint order of RHD
        InterHand2M6_joint_order = [
            41, # left wrist
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            20, #right wrist
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        ]
        
        # 0~41 for RHD
        RHD_joint_order = [0, #left wrist
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, #right wrist
                           22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        
        # Initialize an array with the same shape as InterHand2M6_joint to store the rearranged joint coordinates
        RHD_joint = np.zeros_like(InterHand2M6_joint)
        
        # Rearrange joint coordinates according to the mapping from InterHand2M6_joint_order to RHD_joint_order
        for idx, joint_idx in enumerate(InterHand2M6_joint_order):
            RHD_joint[idx] = InterHand2M6_joint[joint_idx]
        
        return RHD_joint
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        '''
        keypoint_vis21_gt = sample['keypoint_vis21'].to(self.device) # visiable points mask
        index_root_bone_length = sample['keypoint_scale'].to(self.device) #scale length
        keypoint_xyz_root = sample['keypoint_xyz_root'].to(self.device)
        keypoint_uv21_gt = sample['keypoint_uv21'].to(self.device) # uv coordinate
        keypoint_xyz21_gt = sample['keypoint_xyz21'].to(self.device) # xyz absolute coordinate
        keypoint_xyz21_rel_normed_gt = sample['keypoint_xyz21_rel_normed'].to(self.device) ## normalized xyz coordinates
        scoremap = sample['scoremap'].to(self.device) #scale length
        camera_intrinsic_matrix = sample['camera_intrinsic_matrix'].to(self.device)
        gt_hand_mask = sample['right_hand_mask'].to(self.device)

        '''
        
        data = self.datalist[idx]

        img_path = data['img_path']
        bbox = data['bbox']
        joint = data['joint']
        hand_type = data['hand_type']
        hand_type_valid = data['hand_type_valid']
        cam_param = data['cam_param']
        focal = cam_param['focal']
        princpt = cam_param['princpt']

        print('img_path', img_path)
        print('bbox', bbox)
        print('hand_type', hand_type)
        print('hand_type_valid', hand_type_valid)
        print('focal', focal)
        print('princpt', princpt)

        image = cv2.imread(img_path)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        height, width, _ = image.shape
    
        
        if hand_type == 'right' or hand_type == 'left':
            joint_cam = joint['cam_coord'].copy()
            joint_img = joint['img_coord'].copy()
            joint_valid = joint['valid'].copy()


            data_dict = dict()
            keypoint_xyz = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_cam)
            keypoint_uv = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_img)
            keypoint_vis = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_valid)
            camera_intrinsic_matrix = np.array([
                                                [focal[0], 0, princpt[0]],
                                                [0, focal[1], princpt[1]],
                                                [0, 0, 1]
                                            ])
            # 1. Read keypoint xyz
            keypoint_xyz = torch.tensor(keypoint_xyz, dtype=torch.float32)

            # Calculate palm coord
            if not self.use_wrist_coord:
                palm_coord_l = 0.5 * (keypoint_xyz[0, :] + keypoint_xyz[12, :]).unsqueeze(0)
                palm_coord_r = 0.5 * (keypoint_xyz[21, :] + keypoint_xyz[33, :]).unsqueeze(0)
                keypoint_xyz = torch.cat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

            data_dict['keypoint_xyz'] = keypoint_xyz

            # 2. Read keypoint uv
            keypoint_uv = torch.tensor(keypoint_uv, dtype=torch.float32)

            # Calculate palm coord
            if not self.use_wrist_coord:
                palm_coord_uv_l = 0.5 * (keypoint_uv[0, :] + keypoint_uv[12, :]).unsqueeze(0)
                palm_coord_uv_r = 0.5 * (keypoint_uv[21, :] + keypoint_uv[33, :]).unsqueeze(0)
                keypoint_uv = torch.cat([palm_coord_uv_l, keypoint_uv[1:21, :], palm_coord_uv_r, keypoint_uv[-20:, :]], 0)

            if self.coord_uv_noise:
                noise = torch.normal(mean=0.0, std=self.coord_uv_noise_sigma, size=(42, 2))
                keypoint_uv += noise

            data_dict['keypoint_uv'] = keypoint_uv


            # 3. Camera intrinsics
            camera_intrinsic_matrix = torch.tensor(camera_intrinsic_matrix, dtype=torch.float32)
            data_dict['camera_intrinsic_matrix'] = camera_intrinsic_matrix

            # 4. Read image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image_rgb, dtype=torch.float32) / 255.0 - 0.5

            # PyTorch doesn't have a direct equivalent for tf.image.random_hue
            # You might need to use PIL or OpenCV for image augmentation
            image = image.permute(2, 0, 1)
            data_dict['image'] = image

            data_dict['right_hand_mask'] = torch.tensor(mask, dtype=torch.int32)
            # 5. Read mask
            # hand_parts_mask = torch.tensor(mask, dtype=torch.int32)
            # data_dict['hand_parts'] = hand_parts_mask
            # hand_mask = hand_parts_mask > 1
            # bg_mask = ~hand_mask
            # data_dict['hand_mask'] = torch.stack([bg_mask, hand_mask], dim=2).int()

            # 6. Read visibility
            keypoint_vis = torch.tensor(keypoint_vis, dtype=torch.bool)

            # Calculate palm visibility
            if not self.use_wrist_coord:
                palm_vis_l = (keypoint_vis[0] | keypoint_vis[12]).unsqueeze(0)
                palm_vis_r = (keypoint_vis[21] | keypoint_vis[33]).unsqueeze(0)
                keypoint_vis = torch.cat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)
            data_dict['keypoint_vis'] = keypoint_vis

            # """ DEPENDENT DATA ITEMS: SUBSET of 21 keypoints"""
            # # figure out dominant hand by analysis of the segmentation mask
            # # Determine dominant hand by analyzing the segmentation mask
            # one_map = torch.ones_like(hand_parts_mask)
            # zero_map = torch.zeros_like(hand_parts_mask)
            # cond_l = (hand_parts_mask > one_map) & (hand_parts_mask < one_map * 18)
            # cond_r = hand_parts_mask > one_map * 17
            # hand_map_l = torch.where(cond_l, one_map, zero_map)
            # hand_map_r = torch.where(cond_r, one_map, zero_map)

            # data_dict['hand_map_l'] = hand_map_l
            # data_dict['hand_map_r'] = hand_map_r

            # num_px_left_hand = hand_map_l.sum()
            # num_px_right_hand = hand_map_r.sum()
            # # print('num_px_left_hand.shape', num_px_left_hand.shape)
            # # print('num_px_right_hand.shape', num_px_right_hand.shape)
            # # print(f'num_px_left_hand:{num_px_left_hand}, num_px_right_hand:{num_px_right_hand}')
            # if img_name == '00028.png':
            #     # plot_mask_on_image(hand_map_l, hand_map_r, image_rgb.copy())
            #     pass
            # Produce the 21 subset using the segmentation masks
            # We only deal with the more prominent hand for each frame and discard the second set of keypoints
            keypoint_xyz_left = keypoint_xyz[:21, :]
            keypoint_xyz_right = keypoint_xyz[-21:, :]
            
            cond_left = torch.tensor(True) if hand_type == 'left' else torch.tensor(False)
            # print('hand_map_l.shape', hand_map_l.shape) #torch.Size([320, 320])
            # print('hand_map_r.shape', hand_map_r.shape) #torch.Size([320, 320])
            # print('cond_left', cond_left) #tensor(True)
            # if cond_left.item():
            #     data_dict['right_hand_mask'] =  torch.flip(hand_map_l, dims=[1])
            # else:
            #     data_dict['right_hand_mask'] = hand_map_r

            # if img_name == '00028.png':
            #     cond_left = torch.tensor(False)
            # #     pass
            # # print('cond_left', cond_left)

            hand_side = torch.where(cond_left, torch.tensor(0), torch.tensor(1))
            # print('hand_side', hand_side)

            cond_left = cond_left.repeat(keypoint_xyz_left.shape[0], keypoint_xyz_left.shape[1])
            # print('cond_left.shape', cond_left.shape)
            # print('keypoint_xyz_left.shape', keypoint_xyz_left.shape)
            # print('keypoint_xyz_right.shape', keypoint_xyz_right.shape)
            
            # Create a tensor of the same shape as keypoint_xyz_left, filled with cond_left
            # Now use this expanded condition in torch.where
            keypoint_xyz21 = torch.where(cond_left, keypoint_xyz_left, keypoint_xyz_right)
            # print('keypoint_xyz21.shape', keypoint_xyz21.shape)

            # hand_side = torch.tensor(0)
            data_dict['hand_side'] = F.one_hot(hand_side, num_classes=2).float()

            # Invert the X-axis coordinates if it is the left hand
            keypoint_xyz21 = torch.where(hand_side == 0, torch.cat([-keypoint_xyz21[..., :1], keypoint_xyz21[..., 1:]], dim=-1), keypoint_xyz21)
            data_dict['keypoint_xyz21'] = keypoint_xyz21

            # Make coords relative to root joint
            keypoint_xyz_root = keypoint_xyz21[0, :]  # this is the palm coord
            keypoint_xyz21_rel = keypoint_xyz21 - keypoint_xyz_root # relative coords in metric coords
            if not self.use_wrist_coord:
                index_root_bone_length = torch.sqrt((keypoint_xyz21_rel[12, :] - keypoint_xyz21_rel[11, :]).pow(2).sum())
            else:
                index_root_bone_length = torch.sqrt((keypoint_xyz21_rel[12, :]).pow(2).sum())

            # print('index_root_bone_length', index_root_bone_length)
            data_dict['keypoint_scale'] = index_root_bone_length.unsqueeze(0)
            data_dict['keypoint_xyz21_rel_normed'] = keypoint_xyz21_rel / index_root_bone_length ##normalized by length of 12->0
            data_dict['keypoint_xyz_root'] = keypoint_xyz_root

            # Calculate local coordinates
            # Assuming bone_rel_trafo is a defined function compatible with PyTorch
            keypoint_xyz21_local = bone_rel_trafo(data_dict['keypoint_xyz21_rel_normed'])
            data_dict['keypoint_xyz21_local'] = keypoint_xyz21_local.squeeze() # 


            # calculate viewpoint and coords in canonical coordinates
            kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(data_dict['keypoint_xyz21_rel_normed'])
            kp_coord_xyz21_rel_can, rot_mat = torch.squeeze(kp_coord_xyz21_rel_can), torch.squeeze(rot_mat)
            # kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, torch.logical_not(cond_left))
            data_dict['kp_coord_xyz21_rel_can'] = kp_coord_xyz21_rel_can
            data_dict['rot_mat'] = torch.inverse(rot_mat)
            
            # Set of 21 for visibility
            keypoint_vis_left = keypoint_vis[:21]
            keypoint_vis_right = keypoint_vis[-21:]
            # print('keypoint_vis_left.shape', keypoint_vis_left.shape)
            # print('keypoint_vis_right.shape', keypoint_vis_right.shape)        
            keypoint_vis21 = torch.where(cond_left[:, 0:1], keypoint_vis_left, keypoint_vis_right)
            data_dict['keypoint_vis21'] = keypoint_vis21
            # print('keypoint_vis21.shape', keypoint_vis21.shape)

            # Set of 21 for UV coordinates
            keypoint_uv_left = keypoint_uv[:21, :]
            keypoint_uv_right = keypoint_uv[-21:, :]
            keypoint_uv21 = torch.where(cond_left[:,:2], keypoint_uv_left, keypoint_uv_right)
            # print('keypoint_uv21.shape', keypoint_uv21.shape)
            # print('keypoint_uv_left.shape', keypoint_uv_left.shape)
            # print('keypoint_uv_right.shape', keypoint_uv_right.shape)
            # print('cond_left[:,:2].shape', cond_left[:,:2].shape)
            # if img_name == '00028.png':
            #     plot_uv_on_image(keypoint_uv21.numpy(), (255*(0.5+image.permute(1, 2, 0))).numpy().astype(np.uint8), keypoint_vis21.numpy().squeeze())
                # plot_uv_on_image(keypoint_uv.numpy(), (255*(0.5+image.permute(1, 2, 0))).numpy().astype(np.uint8), keypoint_vis.numpy().squeeze())

            data_dict['keypoint_uv21'] = keypoint_uv21
            # if img_name == '00028.png':
            # #     # print('keypoint_vis21', keypoint_vis21)
            #     print('1 keypoint_uv21', keypoint_uv21)

            '''
            Assuming hand_ Side 0 represents left hand, 1 represents right hand
            # Mirror transformation on the U coordinate for the left hand. Mirror the whole image for flipping the left hand horizontally.
            '''
            image = torch.where(hand_side == 0, image.flip(dims=[2]), image)
            data_dict['image'] = image

            mirrored_u = torch.where(hand_side == 0, width - keypoint_uv21[:, 0], keypoint_uv21[:, 0])        
            keypoint_uv21 = torch.cat([mirrored_u.unsqueeze(1), keypoint_uv21[:, 1:2]], dim=1)
            data_dict['keypoint_uv21'] = keypoint_uv21


            """ DEPENDENT DATA ITEMS: HAND CROP """
            if self.hand_crop:
                keypoint_uv21 = data_dict['keypoint_uv21']
                # Re-importing torch after code execution state reset

                # Assuming keypoint_uv21 is defined with shape [21, 2]
                # For demonstration, let's define a sample tensor that might contain some keypoints outside the valid range

                # Filter keypoints to include only those with values > 0 and < 320 for both dimensions
                valid_keypoints = keypoint_uv21[(keypoint_uv21[:, 0] > 0) & (keypoint_uv21[:, 0] < width) & 
                                                (keypoint_uv21[:, 1] > 0) & (keypoint_uv21[:, 1] < height)]

                # Calculate the center of valid keypoints
                if valid_keypoints.shape[0] > 0:  # Check if there are any valid keypoints
                    crop_center = valid_keypoints.mean(dim=0)
                else:
                    crop_center = torch.tensor([self.crop_size/2, self.crop_size/2])  # Default or fallback value if no valid keypoints

                crop_center_flipped = crop_center[[1, 0]]  # Flip dimensions to match the original request


                crop_center = crop_center_flipped.view(2)
                # print(f'crop_center: {crop_center}')

                if self.crop_center_noise:
                    noise = torch.normal(mean=0.0, std=self.crop_center_noise_sigma, size=(2,))
                    crop_center += noise

                crop_scale_noise = 1.0
                if self.crop_scale_noise:
                    crop_scale_noise = torch.rand(1).item() * 0.2 + 1.0  # 在 1.0 到 1.2 之间

                # select visible coords only
                keypoint_h = keypoint_uv21[:, 1][keypoint_vis21.squeeze()]
                keypoint_w = keypoint_uv21[:, 0][keypoint_vis21.squeeze()]
                keypoint_hw = torch.stack([keypoint_h, keypoint_w], dim=1)


                # determine size of crop (measure spatial extend of hw coords first)
                if keypoint_hw.nelement() == 0:
                    min_coord = torch.tensor(0.0)
                    max_coord = torch.tensor(self.image_size)
                else:
                    min_coord = torch.maximum(torch.min(keypoint_hw, dim=0)[0], torch.tensor(0.0))
                    max_coord = torch.minimum(torch.max(keypoint_hw, dim=0)[0], torch.tensor(self.image_size))
                # print(f'min_coord: {min_coord}, max_coord: {max_coord}')
                # find out larger distance wrt the center of crop
                crop_size_best = 2 * torch.maximum(max_coord - crop_center, crop_center - min_coord) + 20 ### here, 20 is for margin
                # print(f'crop_size_best: {crop_size_best}')

                crop_size_best = torch.max(crop_size_best)
                crop_size_best = torch.clamp(crop_size_best, min=50.0, max=500.0)
                # print(f'crop_size_best: {crop_size_best}')

                # catch problem, when no valid kp available
                if not torch.all(torch.isfinite(crop_size_best)):
                    crop_size_best = torch.tensor(200.0)


                # calculate necessary scaling
                scale = self.crop_size / crop_size_best
                scale = torch.clamp(scale, min=1.0, max=10.0) * crop_scale_noise
                # data_dict['crop_scale'] = scale
                # print(f'scale: {scale}')


                if self.crop_offset_noise:
                    noise = torch.normal(mean=0.0, std=self.crop_offset_noise_sigma, size=(2,))
                    crop_center += noise

                # Crop image
                # img_crop = crop_image_from_xy_torch(torch.unsqueeze(image, 0), crop_center.unsqueeze(0), self.crop_size, scale)
                crop_size_scaled = int(self.crop_size / scale)

                # Calculate crop coordinates
                # y1 = int(crop_center[0] - crop_size_scaled // 2) if int(crop_center[0] - crop_size_scaled // 2) > 0 else 0
                # y2 = y1 + crop_size_scaled if y1 + crop_size_scaled < height else height
                

                # x1 = int(crop_center[1] - crop_size_scaled // 2) if int(crop_center[1] - crop_size_scaled // 2) > 0 else 0
                # x2 = x1 + crop_size_scaled if x1 + crop_size_scaled < width else width

                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                
                # print(f'y1:{y1} ~ y2:{y2}; x1:{x1} ~ x2:{x2}, scale_y:{scale_y}, scale_x:{scale_x}')

                length_y = y2 - y1
                scale_y = self.crop_size / length_y
                length_x = x2 - x1
                scale_x = self.crop_size / length_x

                # Crop and resize
                cropped_img = image[:, y1:y2, x1:x2]
                # print('cropped_img.shape', cropped_img.shape)
                cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
                cropped_img = cropped_img.squeeze(0)
                
                right_hand_mask = data_dict['right_hand_mask'] 
                right_hand_mask = right_hand_mask[y1:y2, x1:x2]
                offset = 10
                right_hand_mask[offset:-offset, offset:-offset] = 1
                # print('cropped_img.shape', cropped_img.shape)

                right_hand_mask = F.interpolate(
                    right_hand_mask.unsqueeze(0).unsqueeze(0).float(),  # Convert to float
                    size=(self.crop_size, self.crop_size), 
                    mode='nearest'
                )
                right_hand_mask = right_hand_mask.squeeze(0).squeeze(0)
                
                data_dict['right_hand_mask'] = (right_hand_mask>0).float()
                # print('data_dict[image_crop].shape', data_dict['image_crop'].shape)

                # Modify uv21 coordinates
                keypoint_uv21_u = (keypoint_uv21[:, 0] - x1) * scale_x
                keypoint_uv21_v = (keypoint_uv21[:, 1] - y1) * scale_y
                keypoint_uv21 = torch.stack([keypoint_uv21_u, keypoint_uv21_v], dim=1)
                data_dict['keypoint_uv21'] = keypoint_uv21

                # if hand_side.item() == 0:# left hand
                #     cropped_img = cropped_img.flip(dims=[2])
                #     new_keypoint_uv21[:, 0] = width - new_keypoint_uv21[:, 0]
                data_dict['image_crop'] = cropped_img
                # if img_name == '00028.png':
                #     plot_uv_on_image(new_keypoint_uv21.numpy(), (255*(0.5+cropped_img.permute(1, 2, 0))).numpy().astype(np.uint8), keypoint_vis21.numpy().squeeze())
                #     print(f'new_keypoint_uv21: {new_keypoint_uv21}')

                #     pass

                # Modify camera intrinsics
                scale = scale.view(1, -1)
                scale_matrix = torch.tensor([[scale_x, 0.0, 0.0],
                                            [0.0, scale_y, 0.0],
                                            [0.0, 0.0, 1.0]], dtype=torch.float32)

                trans1 = x1* scale_x
                trans2 = y1* scale_y
                trans_matrix = torch.tensor([[1.0, 0.0, -trans1],
                                            [0.0, 1.0, -trans2],
                                            [0.0, 0.0, 1.0]], dtype=torch.float32)

                
                camera_intrinsic_matrix = torch.matmul(trans_matrix, torch.matmul(scale_matrix, camera_intrinsic_matrix))
                data_dict['camera_intrinsic_matrix'] = camera_intrinsic_matrix
                # print('camera_intrinsic_matrix.shape', camera_intrinsic_matrix.shape)
                # new_pro_uv21 = camera_xyz_to_uv(keypoint_xyz21, camera_intrinsic_matrix)
                # if img_name == '00028.png':
                #     plot_uv_on_image(new_pro_uv21.numpy(), (255*(0.5+cropped_img.permute(1, 2, 0))).numpy().astype(np.uint8), keypoint_vis21.numpy().squeeze())
                #     pass
                #     print(f'new_pro_uv21: {new_pro_uv21}')
                
            # print('keypoint_uv21.shape',keypoint_uv21.shape) # torch.Size([21, 2])
            ### DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints
            if self.calculate_scoremap:
                keypoint_hw21 = torch.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], dim=-1) # create scoremaps from the subset of 2D annoataion
                scoremap_size = self.image_size

                if self.hand_crop:
                    scoremap_size = (self.crop_size, self.crop_size)

                scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                            scoremap_size,
                                                            self.sigma,
                                                            valid_vec=keypoint_vis21)


                if self.scoremap_dropout:
                    # Apply dropout to scoremap
                    scoremap = F.dropout(scoremap, p=self.scoremap_dropout_prob, training=self.training)
                    scoremap *= self.scoremap_dropout_prob
                
                # print('scoremap.shape', scoremap.shape)
                scoremap = scoremap.permute(2, 0, 1)
                # print('scoremap.shape', scoremap.shape)
                data_dict['scoremap'] = scoremap

            if self.scale_to_size:
                # Resize image
                image, keypoint_uv21, keypoint_vis21 = data_dict['image'], data_dict['keypoint_uv21'], data_dict['keypoint_vis21']

                # Resize the image to the target size
                resize_transform = transforms.Resize(self.scale_target_size)
                image = resize_transform(image)

                # Calculate the scale factors for the keypoints
                s = torch.tensor([image.shape[1], image.shape[2]])  # Assuming image is in CxHxW format
                scale = torch.tensor(self.scale_target_size).float() / s.float()

                # Adjust the keypoint coordinates
                keypoint_uv21 = torch.stack([
                    keypoint_uv21[:, 0] * scale[1],
                    keypoint_uv21[:, 1] * scale[0]
                ], dim=1)

                # Update the data dictionary with the resized image and adjusted keypoints
                data_dict = {
                    'image': image,
                    'keypoint_uv21': keypoint_uv21,
                    'keypoint_vis21': keypoint_vis21
                }


            elif self.random_crop_to_size:
                pass
                # Concatenate image, hand_parts, and hand_mask along the channel dimension
                # Concatenate tensors along the channel dimension
            
                # tensor_stack = torch.cat([data_dict['image'],
                #                         data_dict['hand_parts'].unsqueeze(-1).float(),
                #                         data_dict['hand_mask'].float()], dim=-1)

                # # Get the shape of the stacked tensor
                # s = tensor_stack.shape

                # # Define a RandomCrop transform
                # random_crop = transforms.RandomCrop((self.random_crop_size, self.random_crop_size))

                # # Apply random crop
                # tensor_stack_cropped = random_crop(tensor_stack)

                # # Split the cropped tensor back into image, hand_parts, and hand_mask
                # data_dict = dict()
                # data_dict['image'] = tensor_stack_cropped[:, :, :3]
                # data_dict['hand_parts'] = tensor_stack_cropped[:, :, 3].long()  # Assuming hand_parts needs to be a long tensor
                # data_dict['hand_mask'] = tensor_stack_cropped[:, :, 4:].long()  # Assuming hand_mask needs to be a long tensor

            if config.model_name == 'MANO3DHandPose' or config.joint_order_switched:
                keypoint_vis21 = data_dict['keypoint_vis21']
                keypoint_vis21 = self.switch_joint_order(keypoint_vis21)

                keypoint_uv21 = data_dict['keypoint_uv21']
                keypoint_uv21 = self.switch_joint_order(keypoint_uv21)
                
                keypoint_xyz21 = data_dict['keypoint_xyz21']
                keypoint_xyz21 = self.switch_joint_order(keypoint_xyz21)

                data_dict['keypoint_vis21'] = keypoint_vis21
                data_dict['keypoint_uv21'] = keypoint_uv21
                data_dict['keypoint_xyz21'] = keypoint_xyz21

                config.joint_order_switched = True

            data_dict['img_name'] = img_name
            names, tensors = zip(*data_dict.items())
            # print(f'cost {time.time() - start_time:.2f}') #0.02
            return dict(zip(names, tensors))



    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        # print('valid_vec.shape', valid_vec.shape)

        sigma = torch.tensor(sigma, dtype=torch.float32)
        assert len(output_size) == 2
        coords_uv = coords_uv.to(torch.int32)

        if valid_vec is not None:
            valid_vec = valid_vec.to(torch.float32)
            valid_vec = torch.squeeze(valid_vec)
            cond_val = valid_vec > 0.5
        else:
            cond_val = torch.ones_like(coords_uv[:, 0], dtype=torch.float32) > 0.5

        cond_1_in = (coords_uv[:, 0] < output_size[0]-1) & (coords_uv[:, 0] > 0)
        cond_2_in = (coords_uv[:, 1] < output_size[1]-1) & (coords_uv[:, 1] > 0)
        cond_in = cond_1_in & cond_2_in
        # print('cond_val.shape', cond_val.shape)
        # print('cond_in.shape', cond_in.shape)
        cond = cond_val & cond_in

        coords_uv = coords_uv.to(torch.float32)


        # create meshgrid
        X, Y = torch.meshgrid(torch.arange(output_size[0], dtype=torch.float32), 
                              torch.arange(output_size[1], dtype=torch.float32), indexing='ij')

        X = torch.unsqueeze(X, -1)
        Y = torch.unsqueeze(Y, -1)

        # Replicate X and Y for each keypoint
        X_b = X.repeat(1, 1, coords_uv.shape[0]) - coords_uv[:, 0].view(1, 1, -1)
        Y_b = Y.repeat(1, 1, coords_uv.shape[0]) - coords_uv[:, 1].view(1, 1, -1)

        # Calculate squared distances
        dist = X_b.pow(2) + Y_b.pow(2)

        # Generate scoremap
        scoremap = torch.exp(-dist / sigma.pow(2)) * cond.view(1, 1, -1).float()


        return scoremap


    def switch_joint_order(self, joints):
        for i in range(1, 21, 4):
            joints[[i, i + 3]] = joints[[i + 3, i]]
            joints[[i + 1, i + 2]] = joints[[i + 2, i + 1]]
        return joints



if __name__ == '__main__':
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Dataset(transform, 'val')
    batch_size = 1
    num_workers = 0
    # Creating the DataLoader
    shuffle = True
    # shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)


    i = 0
    interaction_i = 0
    for batch in dataloader:

        inputs, targets, meta_info, _ = batch
        '''        inputs = {'img': img}
        targets = {'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'inv_trans': inv_trans, 'hand_type_valid': 1}
        '''
        print(inputs['img'].shape) # torch.Size([bs, 3, 256, 256])
        print(targets['joint_coord'].shape) # torch.Size([bs, 42, 3])
        print(targets['rel_root_depth'].shape) # torch.Size([bs, 1])
        print(targets['hand_type'].shape) # torch.Size([bs, 2])
        print(meta_info['joint_valid'].shape) # torch.Size([bs, 42])
        print(meta_info['root_valid'].shape) # torch.Size([bs, 1])
        print(meta_info['inv_trans'].shape) # torch.Size([bs, 2, 3])
        print(meta_info['hand_type_valid'].shape) # torch.Size([bs, 1])

        img = (inputs['img'].cpu().numpy()*255).astype(np.uint8)
        img_path = inputs['img_path']
        img_name = img_path[0].split('/')[-1]
        shutil.copy(img_path[0], f'img_examples/{img_name}')
        cv2.imwrite(f"img_examples/{img_name.split('.')[0]}_crop.{img_name.split('.')[1]}", img[0].transpose(1,2,0)[:,:,::-1])
        # break
        i += 1
        print('')
        print('')
        # if i > 6:
        #     break
        if 'ROM02_Interaction_2_Hand' in img_path:
            interaction_i += 1
            if i > 3:
                break
            
            
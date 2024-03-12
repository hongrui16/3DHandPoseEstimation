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

class InterHand2M6Dataset(Dataset):

    def __init__(self, root_dir = None, set_type = None, fast_trainval=False):

        assert set_type in ('train', 'test', 'val')
        mode = set_type
        self.mode = set_type # train, test, val
        self.dataset_root_dir = '/scratch/rhong5/dataset/InterHand/InterHand2.6M'
        self.img_path = f'{self.dataset_root_dir}/images'
        self.annot_path =  f'{self.dataset_root_dir}/annotations'
        if self.mode == 'val':
            self.rootnet_output_path =  f'{self.dataset_root_dir}/rootnet_output/rootnet_interhand2.6m_output_val.json'
        elif self.mode == 'test':
            self.rootnet_output_path =  f'{self.dataset_root_dir}/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.rootnet_output_dir =  f'{self.dataset_root_dir}/rootnet_output'
        os.makedirs(self.rootnet_output_dir, exist_ok=True)

        # self.transform = transform
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
            
            if hand_type == 'interacting':
                continue

            if fast_trainval and mode == 'train' and len(self.datalist) >= 8000:
                break
            elif fast_trainval and mode == 'val' and len(self.datalist) >= 1000:
                break
            elif fast_trainval and mode == 'test' and len(self.datalist) >= 1000:
                break

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

        joint_cam = joint['cam_coord'].copy()
        joint_img = joint['img_coord'].copy()
        joint_valid = joint['valid'].copy()

        # print('img_path', img_path)
        # print('bbox', bbox)
        # print('hand_type', hand_type)
        # print('hand_type_valid', hand_type_valid)
        # print('focal', focal)
        # print('princpt', princpt)
        # print('joint_cam', joint_cam)
        # print('joint_img', joint_img)
        # print('joint_valid', joint_valid)

        image = cv2.imread(img_path)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        height, width, _ = image.shape
        bbox = np.array(bbox, dtype=np.int32)
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        bbox[2] = width if bbox[0] + bbox[2] > width else bbox[2]
        bbox[3] = height if bbox[1] + bbox[3] > height else bbox[3]
        
        


        data_dict = dict()
        keypoint_xyz = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_cam)
        keypoint_xyz /= 1000.0  # mm to meters
        keypoint_uv = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_img)
        keypoint_vis = self.convert_joint_order_from_InterHand2M6_to_RHD(joint_valid)
        camera_intrinsic_matrix = np.array([
                                            [focal[0], 0, princpt[0]],
                                            [0, focal[1], princpt[1]],
                                            [0, 0, 1]
                                        ])
        keypoint_xyz = torch.tensor(keypoint_xyz, dtype=torch.float32)
        keypoint_uv = torch.tensor(keypoint_uv, dtype=torch.int32)
        keypoint_vis = torch.tensor(keypoint_vis, dtype=torch.bool)
        keypoint_vis = keypoint_vis.unsqueeze(1)
        
        # 1. Read keypoint xyz
        # Calculate palm coord
        if not self.use_wrist_coord:
            palm_coord_l = 0.5 * (keypoint_xyz[0, :] + keypoint_xyz[12, :]).unsqueeze(0)
            palm_coord_r = 0.5 * (keypoint_xyz[21, :] + keypoint_xyz[33, :]).unsqueeze(0)
            keypoint_xyz = torch.cat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

        data_dict['keypoint_xyz'] = keypoint_xyz

        # 2. Read keypoint uv
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

        # 5. Read mask
        hand_parts_mask = torch.tensor(mask, dtype=torch.int32)
        data_dict['hand_parts'] = hand_parts_mask
        data_dict['hand_mask'] = hand_parts_mask
        data_dict['right_hand_mask'] = hand_parts_mask

        # hand_mask = hand_parts_mask > 1
        # bg_mask = ~hand_mask
        # data_dict['hand_mask'] = torch.stack([bg_mask, hand_mask], dim=2).int()

        # 6. Read visibility
        # print('keypoint_vis', keypoint_vis.shape)

        # Calculate palm visibility
        if not self.use_wrist_coord:
            palm_vis_l = (keypoint_vis[0] | keypoint_vis[12]).unsqueeze(0)
            palm_vis_r = (keypoint_vis[21] | keypoint_vis[33]).unsqueeze(0)
            keypoint_vis = torch.cat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)
        # print('keypoint_vis', keypoint_vis.shape)
        data_dict['keypoint_vis'] = keypoint_vis

        
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
        data_dict['keypoint_xyz21_rel'] = keypoint_xyz21_rel

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
            
            x1, y1, w, h = bbox.tolist()
            x2 = x1 + w
            y2 = y1 + h
            
            # print(f'y1:{y1} ~ y2:{y2}; x1:{x1} ~ x2:{x2}, scale_y:{scale_y}, scale_x:{scale_x}')

            scale_y = self.crop_size / h
            scale_x = self.crop_size / w

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
            # scale = scale.view(1, -1)
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

        data_dict['img_name'] = img_path.split('/')[-1]
        names, tensors = zip(*data_dict.items())
        # print(f'cost {time.time() - start_time:.2f}') #0.02
        return dict(zip(names, tensors))



    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        # print('coords_uv.shape', coords_uv.shape)
        # print('output_size', output_size)
        # print('coords_uv.shape', coords_uv.shape)
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
    dataset = DataloaderInterHand2M6(transform, 'val', fast_debug = True)
    batch_size = 1
    num_workers = 0
    # Creating the DataLoader
    shuffle = True
    # shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)


    i = 0
    interaction_i = 0
    for batch in dataloader:

        scoremap = batch['scoremap']
        images = batch['image']
        image_crop = batch['image_crop']

        masks = batch['hand_mask']
        keypoints_uv = batch['keypoint_uv']
        keypoints_uv_visible = batch['keypoint_vis']
        keypoints_xyz = batch['keypoint_xyz']
        # print('keypoints_xyz', keypoints_xyz)

        keypoint_xyz21 = batch['keypoint_xyz21']        
        keypoint_uv21 = batch['keypoint_uv21']
        keypoint_vis21 = batch['keypoint_vis21']

        camera_matrices = batch['camera_intrinsic_matrix']
        index_root_bone_length = batch['keypoint_scale']
        img_name = batch['img_name']
        hand_side = batch['hand_side']

        keypoint_scale = batch['keypoint_scale']
        keypoint_xyz21_rel_normed = batch['keypoint_xyz21_rel_normed']
        keypoint_xyz_root = batch['keypoint_xyz_root']
        hand_side = batch['hand_side']
        right_hand_mask = batch['right_hand_mask']
        # hand_parts = batch['hand_parts']
        # hand_map_l = batch['hand_map_l']
        print('img_name:', img_name)
        # print('keypoints_xyz:', keypoints_xyz)
        # print('keypoint_uv:', keypoints_uv)
        # print('keypoints_uv_visible:', keypoints_uv_visible)
    
        print('keypoints_xyz.shape:', keypoints_xyz.shape) # torch.Size([BS, 42, 3])
        print('keypoint_uv21.shape:', keypoint_uv21.shape) # torch.Size([1, 21, 2])
        print('keypoint_vis21.shape:', keypoint_vis21.shape) # torch.Size([1, 21, 1])
        print('keypoint_xyz21.shape:', keypoint_xyz21.shape) # torch.Size([1, 21, 3])
        print('keypoints_uv.shape:', keypoints_uv.shape) # torch.Size([BS, 42, 2])
        # print('keypoints_uv_visible.shape:', keypoints_uv_visible.shape) # torch.Size([BS, 42, 1])
        print('camera_matrices.shape:', camera_matrices.shape) # torch.Size([BS, 3, 3])
        # print('camera_matrices\n', camera_matrices)
        print('index_root_bone_length.shape:', index_root_bone_length.shape) # torch.Size([BS, 1])
        print('keypoint_xyz_root', keypoint_xyz_root.shape) #torch.Size([bs, 3])

        # print('')
        # # print('keypoints_xyz[:, :3]', keypoints_xyz[:, :3])
        # print('keypoint_xyz21[:, :6]\n', keypoint_xyz21[:, :6])
        # print('keypoint_uv21[:, :3]\n', keypoint_uv21[:, :6])
        # # print('keypoint_vis21[:, :3]', keypoint_vis21[:, :3])
        # print('keypoint_xyz21_rel_normed[:, :3]\n', keypoint_xyz21_rel_normed[:, :3])

        # print('keypoint_xyz21_rel_normed', keypoint_xyz21_rel_normed)
        # print('keypoint_uv21', keypoint_uv21)
        # print('keypoint_scale', keypoint_scale.shape) #torch.Size([bs, 1])
        
        # print('images.shape:', images.shape) # torch.Size([BS, 3, 3])
        # print('image_crop.shape:', image_crop.shape) # torch.Size([BS, 3, 3])
        # print('keypoint_xyz21.shape',keypoint_xyz21.shape)
        # new_pro_uv21 = camera_xyz_to_uv(keypoint_xyz21[0], camera_matrices[0])
        # print('new_pro_uv21.shape',new_pro_uv21.shape)
        # # plot_uv_on_image(new_pro_uv21.numpy(), (255*(0.5+image_crop[0].permute(1, 2, 0))).numpy().astype(np.uint8), keypoint_vis21[0].numpy().squeeze())
        # print('keypoint_uv21[0,:3]',keypoint_uv21[0,:3])
        # print('new_pro_uv21[:3]',new_pro_uv21[:3])

        # kp_coord_xyz21_rel_can = batch['kp_coord_xyz21_rel_can']
        # rot_mat = batch['rot_mat']
        # print('kp_coord_xyz21_rel_can.shape',kp_coord_xyz21_rel_can.shape) # torch.Size([1, 21, 3])
        # print('rot_mat.shape',rot_mat.shape) # torch.Size([1, 3, 3])
        # break
        print(f'i: {i}\n')
        i += 1
        # break
        if i > 8:
            break
    # print('right_hand_mask.shape', right_hand_mask.shape) # torch.Size([1, 256, 256])
    print('scoremap.shape', scoremap.shape) # torch.Size([1, 21, 256, 256])
    scoremap_0th_channel = scoremap[0, 0, :, :].cpu().numpy()
    plt.imshow(scoremap_0th_channel, cmap='gray')  # 'gray' colormap for single-channel visualization
    plt.savefig('scoremap_0th_channel.png')
    
    # right_hand_mask = right_hand_mask.cpu().squeeze().numpy()

    # # Use matplotlib to visualize the 0th channel
    # plt.imshow(right_hand_mask, cmap='gray')  # 'gray' colormap for single-channel visualization
    # plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    # plt.show()


    # hand_parts = hand_parts.cpu().squeeze().numpy()

    # # Use matplotlib to visualize the 0th channel
    # plt.imshow(hand_parts, cmap='gray')  # 'gray' colormap for single-channel visualization
    # plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    # plt.show()

    
    # hand_map_l = hand_map_l.cpu().squeeze().numpy()

    # # Use matplotlib to visualize the 0th channel
    # plt.imshow(hand_map_l, cmap='gray')  # 'gray' colormap for single-channel visualization
    # plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    # plt.show()
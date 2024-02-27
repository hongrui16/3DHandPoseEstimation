
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import os
import numpy as np
import cv2
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import platform
import sys
sys.path.append('../..')  

from config import config

from utils import transformations as tr

from utils.plot_anno import *
from utils.general import crop_image_from_xy_torch
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.relative_trafo import bone_rel_trafo
from utils.coordinate_trans import camera_xyz_to_uv


class RHD_HandKeypointsDataset(Dataset):
    def __init__(self, root_dir, set_type='training', transform=None, debug = False, device = 'cpu'):
        assert set_type in ['evaluation', 'training']

        self.root_dir = root_dir
        self.set_type = set_type
        self.training = True if set_type == "training" else False
        self.transform = transform
        self.debug = debug

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

        # Load annotations
        with open(os.path.join(root_dir, set_type, f'anno_{set_type}.pickle'), 'rb') as file:
            self.annotations = pickle.load(file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image, mask, and depth
        start_time = time.time()
        img_name = f'{idx:05d}.png'
        img_filepath = os.path.join(self.root_dir, self.set_type, 'color', f'{idx:05d}.png')
        # print('img_filepath', img_filepath)
        image = cv2.imread(img_filepath)
        mask = cv2.imread(os.path.join(self.root_dir, self.set_type, 'mask', f'{idx:05d}.png'),  0)
        depth = cv2.imread(os.path.join(self.root_dir, self.set_type, 'depth', f'{idx:05d}.png'))
        
        height, width, _ = image.shape

        # Extract annotations
        anno = self.annotations[idx]
        keypoint_uv = anno['uv_vis'][:, :2] ## u, v coordinates of 42 hand keypoints, pixel
        keypoint_vis = anno['uv_vis'][:, 2:] == 1 # visibility of the keypoints, boolean
        keypoint_xyz = anno['xyz']
        camera_intrinsic_matrix = anno['K']

        if self.debug:
            return {'img_name': img_name,
                    'image': image, 'hand_mask': mask, 'hand_side': None,
                  'keypoint_uv': keypoint_uv, 'keypoint_vis': keypoint_vis,
                  'keypoint_xyz': keypoint_xyz, 'camera_intrinsic_matrix': camera_intrinsic_matrix}

        # if img_name == '00028.png':
        #     print('keypoint_uv', keypoint_uv)
        #     print('keypoint_vis', keypoint_vis)
        #     plot_uv_on_image(keypoint_uv, image[:,:,::-1].copy())


        data_dict = dict()

                
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

        # 5. Read mask
        hand_parts_mask = torch.tensor(mask, dtype=torch.int32)
        data_dict['hand_parts'] = hand_parts_mask
        hand_mask = hand_parts_mask > 1
        bg_mask = ~hand_mask
        data_dict['hand_mask'] = torch.stack([bg_mask, hand_mask], dim=2).int()

        # 6. Read visibility
        keypoint_vis = torch.tensor(keypoint_vis, dtype=torch.bool)

        # Calculate palm visibility
        if not self.use_wrist_coord:
            palm_vis_l = (keypoint_vis[0] | keypoint_vis[12]).unsqueeze(0)
            palm_vis_r = (keypoint_vis[21] | keypoint_vis[33]).unsqueeze(0)
            keypoint_vis = torch.cat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)
        data_dict['keypoint_vis'] = keypoint_vis

        """ DEPENDENT DATA ITEMS: SUBSET of 21 keypoints"""
        # figure out dominant hand by analysis of the segmentation mask
        # Determine dominant hand by analyzing the segmentation mask
        one_map = torch.ones_like(hand_parts_mask)
        zero_map = torch.zeros_like(hand_parts_mask)
        cond_l = (hand_parts_mask > one_map) & (hand_parts_mask < one_map * 18)
        cond_r = hand_parts_mask > one_map * 17
        hand_map_l = torch.where(cond_l, one_map, zero_map)
        hand_map_r = torch.where(cond_r, one_map, zero_map)

        data_dict['hand_map_l'] = hand_map_l
        data_dict['hand_map_r'] = hand_map_r

        num_px_left_hand = hand_map_l.sum()
        num_px_right_hand = hand_map_r.sum()
        # print('num_px_left_hand.shape', num_px_left_hand.shape)
        # print('num_px_right_hand.shape', num_px_right_hand.shape)
        # print(f'num_px_left_hand:{num_px_left_hand}, num_px_right_hand:{num_px_right_hand}')
        if img_name == '00028.png':
            # plot_mask_on_image(hand_map_l, hand_map_r, image_rgb.copy())
            pass
        # Produce the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        keypoint_xyz_left = keypoint_xyz[:21, :]
        keypoint_xyz_right = keypoint_xyz[-21:, :]
        
        cond_left = (num_px_left_hand > num_px_right_hand)
        # print('hand_map_l.shape', hand_map_l.shape) #torch.Size([320, 320])
        # print('hand_map_r.shape', hand_map_r.shape) #torch.Size([320, 320])
        # print('cond_left', cond_left) #tensor(True)
        if cond_left.item():
            data_dict['right_hand_mask'] =  torch.flip(hand_map_l, dims=[1])
        else:
            data_dict['right_hand_mask'] = hand_map_r

        if img_name == '00028.png':
            cond_left = torch.tensor(False)
        #     pass
        # print('cond_left', cond_left)

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
        # index_root_bone_length = torch.sqrt((keypoint_xyz21_rel[12, :] - keypoint_xyz21_rel[11, :]).pow(2).sum())
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
            y1 = int(crop_center[0] - crop_size_scaled // 2) if int(crop_center[0] - crop_size_scaled // 2) > 0 else 0
            y2 = y1 + crop_size_scaled if y1 + crop_size_scaled < height else height
            

            x1 = int(crop_center[1] - crop_size_scaled // 2) if int(crop_center[1] - crop_size_scaled // 2) > 0 else 0
            x2 = x1 + crop_size_scaled if x1 + crop_size_scaled < width else width
            
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
    if platform.system() == 'Windows':
        dataset_dir = '../../dataset/RHD'
    elif platform.system() == 'Linux':
        dataset_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'
    
    
    num_workers = 15
    batch_size=480
    num_workers = 1
    batch_size=1
    gpu_index = None
    cuda_valid = torch.cuda.is_available()
    if cuda_valid:
        print(f"CUDA is available, using GPU {gpu_index}")
        if gpu_index is None:
            device = torch.device(f"cuda")
        else:
            device = torch.device(f"cuda:{gpu_index}")
    else:
        print("CUDA is unavailable, using CPU")
        device = torch.device("cpu")
    transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

    # Creating the dataset
    # dataset = RHD_HandKeypointsDataset(root_dir=dataset_dir, set_type='evaluation', transform=transforms, debug=False)
    dataset = RHD_HandKeypointsDataset(root_dir=dataset_dir, set_type='evaluation', transform=transforms, debug=False)

    # Creating the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    '''
    {'img_name': img_name,
                    'image': image, 'mask': mask, 'depth': depth,
                  'keypoint_uv': keypoint_uv, 'keypoint_vis': keypoint_vis,
                  'keypoint_xyz': keypoint_xyz, 'camera_matrix': camera_intrinsic_matrix}
    '''
    
    
    # Usage example
    i = 0
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
        hand_parts = batch['hand_parts']
        hand_map_l = batch['hand_map_l']
        print('img_name:', img_name)
        # print('keypoints_xyz:', keypoints_xyz)
        # print('keypoint_uv:', keypoints_uv)
        # print('keypoints_uv_visible:', keypoints_uv_visible)
    
        # print('keypoints_xyz.shape:', keypoints_xyz.shape) # torch.Size([BS, 42, 3])
        # print('keypoint_uv21.shape:', keypoint_uv21.shape) # torch.Size([1, 21, 2])
        # print('keypoint_vis21.shape:', keypoint_vis21.shape) # torch.Size([1, 21, 1])
        # print('keypoint_xyz21.shape:', keypoint_xyz21.shape) # torch.Size([1, 21, 3])
        # print('keypoints_uv.shape:', keypoints_uv.shape) # torch.Size([BS, 42, 2])
        # # print('keypoints_uv_visible.shape:', keypoints_uv_visible.shape) # torch.Size([BS, 42, 1])
        # print('camera_matrices.shape:', camera_matrices.shape) # torch.Size([BS, 3, 3])
        # print('camera_matrices\n', camera_matrices)
        # print('index_root_bone_length.shape:', index_root_bone_length.shape) # torch.Size([BS, 1])
        # print('')
        # # print('keypoints_xyz[:, :3]', keypoints_xyz[:, :3])
        # print('keypoint_xyz21[:, :6]\n', keypoint_xyz21[:, :6])
        # print('keypoint_uv21[:, :3]\n', keypoint_uv21[:, :6])
        # # print('keypoint_vis21[:, :3]', keypoint_vis21[:, :3])
        # print('keypoint_xyz21_rel_normed[:, :3]\n', keypoint_xyz21_rel_normed[:, :3])

        # print('keypoint_xyz21_rel_normed', keypoint_xyz21_rel_normed)
        # print('keypoint_uv21', keypoint_uv21)
        # print('keypoint_scale', keypoint_scale)
        # print('keypoint_xyz_root', keypoint_xyz_root, keypoint_xyz_root.shape)
        
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
        # print('kp_coord_xyz21_rel_can.shape',kp_coord_xyz21_rel_can.shape)
        # print('rot_mat.shape',rot_mat.shape)
        # break
        print(f'i: {i}\n')
        i += 1
        break
        if i > 8:
            break
    # print('right_hand_mask.shape', right_hand_mask.shape) # torch.Size([1, 256, 256])
    # scoremap_0th_channel = scoremap[0, :, :, 0].cpu().numpy()
    # plt.imshow(scoremap_0th_channel, cmap='gray')  # 'gray' colormap for single-channel visualization
    
    right_hand_mask = right_hand_mask.cpu().squeeze().numpy()

    # Use matplotlib to visualize the 0th channel
    plt.imshow(right_hand_mask, cmap='gray')  # 'gray' colormap for single-channel visualization
    plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    plt.show()


    hand_parts = hand_parts.cpu().squeeze().numpy()

    # Use matplotlib to visualize the 0th channel
    plt.imshow(hand_parts, cmap='gray')  # 'gray' colormap for single-channel visualization
    plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    plt.show()

    
    hand_map_l = hand_map_l.cpu().squeeze().numpy()

    # Use matplotlib to visualize the 0th channel
    plt.imshow(hand_map_l, cmap='gray')  # 'gray' colormap for single-channel visualization
    plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    plt.show()


    image_crop = (image_crop + 0.5).permute(0, 2, 3, 1).cpu().squeeze().numpy()

    # Use matplotlib to visualize the 0th channel
    plt.imshow(image_crop)  # 'gray' colormap for single-channel visualization
    plt.colorbar()  # Optionally add a colorbar to see the mapping of values to colors
    plt.show()

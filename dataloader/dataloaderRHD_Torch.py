
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import os
import numpy as np
import cv2

import torch.nn.functional as F

import sys
sys.path.append('../')  

from config.config import *
from utils import transformations as tr

from utils.general_torch import crop_image_from_xy_torch
# from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.relative_trafo_torch import bone_rel_trafo


class RHD_HandKeypointsDatasetTorch(Dataset):
    def __init__(self, root_dir, set_type='training', transform=None, debug = False):
        assert set_type in ['evaluation', 'training']

        self.root_dir = root_dir
        self.set_type = set_type
        self.training = True if set_type == "training" else False
        self.transform = transform
        self.debug = debug

        # general parameters
        self.sigma = sigma
        self.shuffle = shuffle
        self.use_wrist_coord = use_wrist_coord
        self.random_crop_to_size = random_crop_to_size
        self.random_crop_size = 256
        self.scale_to_size = scale_to_size
        self.scale_target_size = (240, 320)  # size its scaled down to if scale_to_size=True

        # data augmentation parameters
        self.hue_aug = hue_aug
        self.hue_aug_max = 0.1

        self.hand_crop = hand_crop
        self.coord_uv_noise = coord_uv_noise
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = crop_center_noise
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints

        self.crop_scale_noise = crop_scale_noise
        self.crop_offset_noise = crop_offset_noise
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        self.scoremap_dropout = scoremap_dropout
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


        # Determine dominant hand by analyzing the segmentation mask
        one_map = torch.ones_like(hand_parts_mask)
        zero_map = torch.zeros_like(hand_parts_mask)
        cond_l = (hand_parts_mask > one_map) & (hand_parts_mask < one_map * 18)
        cond_r = hand_parts_mask > one_map * 17
        hand_map_l = torch.where(cond_l, one_map, zero_map)
        hand_map_r = torch.where(cond_r, one_map, zero_map)
        num_px_left_hand = hand_map_l.sum()
        num_px_right_hand = hand_map_r.sum()
        # print('num_px_left_hand.shape', num_px_left_hand.shape)
        # print('num_px_right_hand.shape', num_px_right_hand.shape)

        # Produce the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]
        
        cond_left = (num_px_left_hand > num_px_right_hand)
        

        # Create a tensor of the same shape as kp_coord_xyz_left, filled with cond_left
        cond_left = cond_left.repeat(kp_coord_xyz_left.shape[0], kp_coord_xyz_left.shape[1])
        # print('cond_left.shape', cond_left.shape)
        # print('kp_coord_xyz_left.shape', kp_coord_xyz_left.shape)
        # print('kp_coord_xyz_right.shape', kp_coord_xyz_right.shape)
        # Now use this expanded condition in torch.where
        kp_coord_xyz21 = torch.where(cond_left, kp_coord_xyz_left, kp_coord_xyz_right)
        # print('kp_coord_xyz21.shape', kp_coord_xyz21.shape)



        hand_side = torch.where(num_px_left_hand > num_px_right_hand, torch.tensor(0), torch.tensor(1))
        data_dict['hand_side'] = F.one_hot(hand_side, num_classes=2).float()

        # Invert the X-axis coordinates if it is the left hand
        kp_coord_xyz21 = torch.where(hand_side == 0, torch.cat([-kp_coord_xyz21[..., :1], kp_coord_xyz21[..., 1:]], dim=-1), kp_coord_xyz21)

        data_dict['keypoint_xyz21'] = kp_coord_xyz21

        # Make coords relative to root joint
        kp_coord_xyz_root = kp_coord_xyz21[0, :]  # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root
        index_root_bone_length = torch.sqrt((kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :]).pow(2).sum())
        data_dict['keypoint_scale'] = index_root_bone_length.unsqueeze(-1)
        data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / index_root_bone_length ##normalized by length of 12->11

        # Calculate local coordinates
        # Assuming bone_rel_trafo is a defined function compatible with PyTorch
        kp_coord_xyz21_local = bone_rel_trafo(data_dict['keypoint_xyz21_normed'])
        data_dict['keypoint_xyz21_local'] = kp_coord_xyz21_local.squeeze() # 

        # Handling visibility and UV coordinates
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        # print('keypoint_vis_left.shape', keypoint_vis_left.shape)
        # print('keypoint_vis_right.shape', keypoint_vis_right.shape)
        # print('cond_left[:, 0].shape', cond_left[:, 0].shape)
        
        keypoint_vis21 = torch.where(cond_left[:, 0:1], keypoint_vis_left, keypoint_vis_right)
        data_dict['keypoint_vis21'] = keypoint_vis21
        # print('keypoint_vis21.shape', keypoint_vis21.shape)

        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        keypoint_uv21 = torch.where(cond_left[:,:2], keypoint_uv_left, keypoint_uv_right)
        # print('keypoint_uv21.shape', keypoint_uv21.shape)
        # print('keypoint_uv_left.shape', keypoint_uv_left.shape)
        # print('keypoint_uv_right.shape', keypoint_uv_right.shape)
        # print('cond_left[:,:2].shape', cond_left[:,:2].shape)
        #        #If it is the left hand, perform a mirror transformation on the U coordinate
        #Assuming hand_ Side 0 represents left hand, 1 represents right hand
        # Mirror transformation on the U coordinate for the left hand
        mirrored_u = torch.where(hand_side == 0, width - keypoint_uv21[:, 0], keypoint_uv21[:, 0])
        data_dict['keypoint_uv21'] = torch.cat([mirrored_u.unsqueeze(1), keypoint_uv21[:, 1:2]], dim=1)



        """ DEPENDENT DATA ITEMS: HAND CROP """
        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]

            # catch problem, when no valid kp available (happens almost never)
            if not torch.all(torch.isfinite(crop_center)):
                crop_center = torch.tensor([0.0, 0.0])

            crop_center.set_shape([2, ])

            if self.crop_center_noise:
                noise = torch.normal(mean=0.0, std=self.crop_center_noise_sigma, size=(2,))
                crop_center += noise

            crop_scale_noise = 1.0
            if self.crop_scale_noise:
                crop_scale_noise = torch.rand(1).item() * 0.2 + 1.0  # 在 1.0 到 1.2 之间

            # select visible coords only
            kp_coord_h = keypoint_uv21[:, 1][keypoint_vis21.squeeze()]
            kp_coord_w = keypoint_uv21[:, 0][keypoint_vis21.squeeze()]
            kp_coord_hw = torch.stack([kp_coord_h, kp_coord_w], dim=1)


            # determine size of crop (measure spatial extend of hw coords first)
            min_coord = torch.maximum(torch.min(kp_coord_hw, dim=0)[0], torch.tensor(0.0))
            max_coord = torch.minimum(torch.max(kp_coord_hw, dim=0)[0], torch.tensor(self.image_size))

            # find out larger distance wrt the center of crop
            crop_size_best = 2 * torch.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = torch.max(crop_size_best)
            crop_size_best = torch.clamp(crop_size_best, min=50.0, max=500.0)

            # catch problem, when no valid kp available
            if not torch.all(torch.isfinite(crop_size_best)):
                crop_size_best = torch.tensor(200.0)


            # calculate necessary scaling
            scale = self.crop_size / crop_size_best
            scale = torch.clamp(scale, min=1.0, max=10.0) * crop_scale_noise
            data_dict['crop_scale'] = scale


            if self.crop_offset_noise:
                noise = torch.normal(mean=0.0, std=self.crop_offset_noise_sigma, size=(2,))
                crop_center += noise

            # Crop image
            img_crop = crop_image_from_xy_torch(torch.unsqueeze(image, 0), crop_center, self.crop_size, scale)
            data_dict['image_crop'] = torch.squeeze(img_crop)

            # Modify uv21 coordinates
            crop_center_float = crop_center.float()
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = torch.stack([keypoint_uv21_u, keypoint_uv21_v], dim=1)
            data_dict['keypoint_uv21'] = keypoint_uv21


            # Modify camera intrinsics
            scale = scale.view(1, -1)
            scale_matrix = torch.tensor([[scale, 0.0, 0.0],
                                        [0.0, scale, 0.0],
                                        [0.0, 0.0, 1.0]], dtype=torch.float32)

            crop_center_float = crop_center.float()
            trans1 = crop_center_float[0] * scale - self.crop_size // 2
            trans2 = crop_center_float[1] * scale - self.crop_size // 2
            trans_matrix = torch.tensor([[1.0, 0.0, -trans2],
                                        [0.0, 1.0, -trans1],
                                        [0.0, 0.0, 1.0]], dtype=torch.float32)

            data_dict['camera_intrinsic_matrix'] = torch.matmul(trans_matrix, torch.matmul(scale_matrix, camera_intrinsic_matrix))


        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create scoremaps from the subset of 2D annoataion
        keypoint_hw21 = torch.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], dim=-1)
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
            # Concatenate image, hand_parts, and hand_mask along the channel dimension
            # Concatenate tensors along the channel dimension
            tensor_stack = torch.cat([data_dict['image'],
                                    data_dict['hand_parts'].unsqueeze(-1).float(),
                                    data_dict['hand_mask'].float()], dim=-1)

            # Get the shape of the stacked tensor
            s = tensor_stack.shape

            # Define a RandomCrop transform
            random_crop = transforms.RandomCrop((self.random_crop_size, self.random_crop_size))

            # Apply random crop
            tensor_stack_cropped = random_crop(tensor_stack)

            # Split the cropped tensor back into image, hand_parts, and hand_mask
            data_dict = dict()
            data_dict['image'] = tensor_stack_cropped[:, :, :3]
            data_dict['hand_parts'] = tensor_stack_cropped[:, :, 3].long()  # Assuming hand_parts needs to be a long tensor
            data_dict['hand_mask'] = tensor_stack_cropped[:, :, 4:].long()  # Assuming hand_mask needs to be a long tensor

        
        data_dict['img_name'] = img_name
        names, tensors = zip(*data_dict.items())

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
        x_range = torch.unsqueeze(torch.arange(output_size[0]), 1)
        y_range = torch.unsqueeze(torch.arange(output_size[1]), 0)

        X = torch.tile(x_range, (1, output_size[1])).to(torch.float32)
        Y = torch.tile(y_range, (output_size[0], 1)).to(torch.float32)

        X = torch.unsqueeze(X, -1)
        Y = torch.unsqueeze(Y, -1)

        X_b = X.repeat(1, 1, coords_uv.shape[0])
        Y_b = Y.repeat(1, 1, coords_uv.shape[0])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = torch.square(X_b) + torch.square(Y_b)

        # print('dist.shape', dist.shape)
        # print('sigma.shape', sigma.shape)
        # print('cond.shape', cond.shape)

        scoremap = torch.exp(-dist / torch.square(sigma)) * cond.to(torch.float32)

        return scoremap





if __name__ == '__main__':

    dataset_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'

    transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

    # Creating the dataset
    dataset = RHD_HandKeypointsDatasetTorch(root_dir=dataset_dir, set_type='evaluation', transform=transforms, debug=False)

    # Creating the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    '''
    {'img_name': img_name,
                    'image': image, 'mask': mask, 'depth': depth,
                  'keypoint_uv': keypoint_uv, 'keypoint_vis': keypoint_vis,
                  'kp_coord_xyz': keypoint_xyz, 'camera_matrix': camera_intrinsic_matrix}
    '''
    
    
    # Usage example
    for batch in dataloader:
        images = batch['image']
        masks = batch['hand_mask']
        keypoints_uv = batch['keypoint_uv']
        keypoints_uv_visible = batch['keypoint_vis']
        keypoints_xyz = batch['keypoint_xyz']

        keypoint_xyz21 = batch['keypoint_xyz21']        
        keypoint_uv21 = batch['keypoint_uv21']
        keypoint_vis21 = batch['keypoint_vis21']

        camera_matrices = batch['camera_intrinsic_matrix']
        index_root_bone_length = batch['keypoint_scale']
        img_name = batch['img_name']
        hand_side = batch['hand_side']

        keypoint_scale = batch['keypoint_scale']
        keypoint_xyz21_normed = batch['keypoint_xyz21_normed']

        print('img_name:', img_name)
        # print('keypoints_xyz:', keypoints_xyz)
        # print('kp_coord_uv:', keypoints_uv)
        # print('keypoints_uv_visible:', keypoints_uv_visible)
    
        print('keypoints_xyz.shape:', keypoints_xyz.shape) # torch.Size([BS, 42, 3])
        print('keypoint_uv21.shape:', keypoint_uv21.shape) # torch.Size([BS, 21, 3])
        print('keypoints_uv.shape:', keypoints_uv.shape) # torch.Size([BS, 42, 2])
        # print('keypoints_uv_visible.shape:', keypoints_uv_visible.shape) # torch.Size([BS, 42, 1])
        print('camera_matrices.shape:', camera_matrices.shape) # torch.Size([BS, 3, 3])
        print('camera_matrices', camera_matrices)
        print('index_root_bone_length.shape:', index_root_bone_length.shape) # torch.Size([BS, 1])
        print('')
        print('keypoints_xyz[:, :3]', keypoints_xyz[:, :3])
        print('keypoint_xyz21[:, :3]', keypoint_xyz21[:, :3])
        print('keypoint_uv21[:, :3]', keypoint_uv21[:, :3])
        print('keypoint_vis21[:, :3]', keypoint_vis21[:, :3])
        print('keypoint_scale', keypoint_scale)
        print('keypoint_xyz21_normed[:, :3]', keypoint_xyz21_normed[:, :3])
        
        
        break
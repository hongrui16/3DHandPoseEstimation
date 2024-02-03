
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import os
import numpy as np
import cv2
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()


import sys
sys.path.append('../')  

from config.config import *
from utils import transformations as tr
from utils.general import crop_image_from_xy
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.relative_trafo import bone_rel_trafo


class RHD_HandKeypointsDataset(Dataset):
    def __init__(self, root_dir, set_type='training', transform=None, debug = False):
        assert set_type in ['evaluation', 'training']

        self.root_dir = root_dir
        self.set_type = set_type
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
        image = cv2.imread(os.path.join(self.root_dir, self.set_type, 'color', f'{idx:05d}.png'))
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
        keypoint_xyz = tf.convert_to_tensor(keypoint_xyz, dtype=tf.float32)

        # calculate palm coord
        if not self.use_wrist_coord:
            palm_coord_l = tf.expand_dims(0.5*(keypoint_xyz[0, :] + keypoint_xyz[12, :]), 0)
            palm_coord_r = tf.expand_dims(0.5*(keypoint_xyz[21, :] + keypoint_xyz[33, :]), 0)
            keypoint_xyz = tf.concat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

        data_dict['keypoint_xyz'] = keypoint_xyz

        # 2. Read keypoint uv
        keypoint_uv = tf.convert_to_tensor(keypoint_uv, dtype=tf.float32)

        # calculate palm coord
        if not self.use_wrist_coord:
            palm_coord_uv_l = tf.expand_dims(0.5*(keypoint_uv[0, :] + keypoint_uv[12, :]), 0)
            palm_coord_uv_r = tf.expand_dims(0.5*(keypoint_uv[21, :] + keypoint_uv[33, :]), 0)
            keypoint_uv = tf.concat([palm_coord_uv_l, keypoint_uv[1:21, :], palm_coord_uv_r, keypoint_uv[-20:, :]], 0)

        if self.coord_uv_noise:
            noise = tf.truncated_normal([42, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv += noise

        data_dict['keypoint_uv'] = keypoint_uv

        # 3. Camera intrinsics
        camera_intrinsic_matrix = tf.convert_to_tensor(camera_intrinsic_matrix, dtype=tf.float32)

        data_dict['camera_intrinsic_matrix'] = camera_intrinsic_matrix

        # decode to uint8
        # 4. Read image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image_rgb, dtype=tf.float32)

        # subtract mean
        image = image / 255.0 - 0.5
        if self.hue_aug:
            image = tf.image.random_hue(image, self.hue_aug_max)
        data_dict['image'] = image

        # 5. Read mask
        hand_parts_mask = tf.convert_to_tensor(mask, dtype=tf.int32)

        data_dict['hand_parts'] = hand_parts_mask
        hand_mask = tf.greater(hand_parts_mask, 1)
        bg_mask = tf.logical_not(hand_mask)
        data_dict['hand_mask'] = tf.cast(tf.stack([bg_mask, hand_mask], 2), tf.int32)

        # 6. Read visibilty

        keypoint_vis = tf.cast(keypoint_vis, tf.bool)

        # calculate palm visibility
        if not self.use_wrist_coord:
            palm_vis_l = tf.expand_dims(tf.logical_or(keypoint_vis[0], keypoint_vis[12]), 0)
            palm_vis_r = tf.expand_dims(tf.logical_or(keypoint_vis[21], keypoint_vis[33]), 0)
            keypoint_vis = tf.concat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)
        data_dict['keypoint_vis'] = keypoint_vis


        """ DEPENDENT DATA ITEMS: SUBSET of 21 keypoints"""
        # figure out dominant hand by analysis of the segmentation mask
        one_map, zero_map = tf.ones_like(hand_parts_mask), tf.zeros_like(hand_parts_mask)
        cond_l = tf.logical_and(tf.greater(hand_parts_mask, one_map), tf.less(hand_parts_mask, one_map*18))
        cond_r = tf.greater(hand_parts_mask, one_map*17)
        hand_map_l = tf.where(cond_l, one_map, zero_map)
        hand_map_r = tf.where(cond_r, one_map, zero_map)
        num_px_left_hand = tf.reduce_sum(hand_map_l)
        num_px_right_hand = tf.reduce_sum(hand_map_r)
        # print('num_px_left_hand.shape', num_px_left_hand.shape)
        # print('num_px_right_hand.shape', num_px_right_hand.shape)

        # PRODUCE the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]

        cond_left = tf.logical_and(tf.cast(tf.ones_like(kp_coord_xyz_left), tf.bool), tf.greater(num_px_left_hand, num_px_right_hand))
        kp_coord_xyz21 = tf.where(cond_left, kp_coord_xyz_left, kp_coord_xyz_right)

        # print('cond_left.shape', cond_left.shape)
        # print('kp_coord_xyz_left.shape', kp_coord_xyz_left.shape)
        # print('kp_coord_xyz_right.shape', kp_coord_xyz_right.shape)

        hand_side = tf.where(tf.greater(num_px_left_hand, num_px_right_hand),
                             tf.constant(0, dtype=tf.int32),
                             tf.constant(1, dtype=tf.int32))  # left hand = 0; right hand = 1
        data_dict['hand_side'] = tf.one_hot(hand_side, depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)


        ### If it is the left hand, invert the X-axis coordinates
        kp_coord_xyz21 = tf.where(tf.equal(hand_side, 0),
                          tf.concat([-kp_coord_xyz21[..., :1], kp_coord_xyz21[..., 1:]], axis=-1),
                          kp_coord_xyz21)


        data_dict['keypoint_xyz21'] = kp_coord_xyz21


        # make coords relative to root joint
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        index_root_bone_length = tf.sqrt(tf.reduce_sum(tf.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        data_dict['keypoint_scale'] = index_root_bone_length
        data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / index_root_bone_length  # normalized by length of 12->11

        # calculate local coordinates
        kp_coord_xyz21_local = bone_rel_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_local = tf.squeeze(kp_coord_xyz21_local)
        data_dict['keypoint_xyz21_local'] = kp_coord_xyz21_local

        '''
        the following part seems has problem, and needs to be verified
        # calculate viewpoint and coords in canonical coordinates
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_rel_can, rot_mat = tf.squeeze(kp_coord_xyz21_rel_can), tf.squeeze(rot_mat)
        kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, tf.logical_not(cond_left))
        data_dict['keypoint_xyz21_can'] = kp_coord_xyz21_rel_can
        data_dict['rot_mat'] = tf.matrix_inverse(rot_mat)

        '''
        
        # Set of 21 for visibility
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        # print('keypoint_vis_left.shape', keypoint_vis_left.shape)
        # print('keypoint_vis_right.shape', keypoint_vis_right.shape)
        # print('cond_left[:, 0].shape', cond_left[:, 0].shape)
        keypoint_vis21 = tf.where(cond_left[:, 0], keypoint_vis_left, keypoint_vis_right)
        data_dict['keypoint_vis21'] = keypoint_vis21
        # print('keypoint_vis21.shape', keypoint_vis21.shape)

        # Set of 21 for UV coordinates
        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        keypoint_uv21 = tf.where(cond_left[:, :2], keypoint_uv_left, keypoint_uv_right)
        # print('keypoint_uv21.shape', keypoint_uv21.shape)
        # print('keypoint_uv_left.shape', keypoint_uv_left.shape)
        # print('keypoint_uv_right.shape', keypoint_uv_right.shape)
        # print('cond_left[:,:2].shape', cond_left[:,:2].shape)

        ### new added
        #If it is the left hand, perform a mirror transformation on the U coordinate
        #Assuming hand_ Side 0 represents left hand, 1 represents right hand
        mirrored_u = tf.where(tf.equal(hand_side, 0), width - keypoint_uv21[:, 0], keypoint_uv21[:, 0])
        # update the keypoint_uv21 in the data_dict
        data_dict['keypoint_uv21'] = tf.concat([mirrored_u[:, tf.newaxis], keypoint_uv21[:, 1:2]], axis=1)

        

        """ DEPENDENT DATA ITEMS: HAND CROP """
        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]

            # catch problem, when no valid kp available (happens almost never)
            crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                                  lambda: tf.constant([0.0, 0.0]))
            crop_center.set_shape([2, ])

            if self.crop_center_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_center_noise_sigma)
                crop_center += noise

            crop_scale_noise = tf.constant(1.0)
            if self.crop_scale_noise:
                    crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=1.0, maxval=1.2))

            # select visible coords only
            kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
            kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
            kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

            # determine size of crop (measure spatial extend of hw coords first)
            min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
            max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), self.image_size)

            # find out larger distance wrt the center of crop
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = tf.reduce_max(crop_size_best)
            crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)

            # catch problem, when no valid kp available
            crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                  lambda: tf.constant(200.0))
            crop_size_best.set_shape([])

            # calculate necessary scaling
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            scale *= crop_scale_noise
            data_dict['crop_scale'] = scale

            if self.crop_offset_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                crop_center += noise

            # Crop image
            img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
            data_dict['image_crop'] = tf.squeeze(img_crop)

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            data_dict['keypoint_uv21'] = keypoint_uv21

            # Modify camera intrinsics
            scale = tf.reshape(scale, [1, ])
            scale_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [scale, [0.0], [0.0],
                                                               [0.0], scale, [0.0],
                                                               [0.0], [0.0], [1.0]])
            scale_matrix = tf.reshape(scale_matrix, [3, 3])

            crop_center_float = tf.cast(crop_center, tf.float32)
            trans1 = crop_center_float[0] * scale - self.crop_size // 2
            trans2 = crop_center_float[1] * scale - self.crop_size // 2
            trans1 = tf.reshape(trans1, [1, ])
            trans2 = tf.reshape(trans2, [1, ])
            trans_matrix = tf.dynamic_stitch([[0], [1], [2],
                                              [3], [4], [5],
                                              [6], [7], [8]], [[1.0], [0.0], -trans2,
                                                               [0.0], [1.0], -trans1,
                                                               [0.0], [0.0], [1.0]])
            trans_matrix = tf.reshape(trans_matrix, [3, 3])

            data_dict['camera_intrinsic_matrix'] = tf.matmul(trans_matrix, tf.matmul(scale_matrix, camera_intrinsic_matrix))

        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create scoremaps from the subset of 2D annoataion
        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        scoremap_size = self.image_size
        
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                     scoremap_size,
                                                     self.sigma,
                                                     valid_vec=keypoint_vis21)
        
        if self.scoremap_dropout:
            scoremap = tf.nn.dropout(scoremap, self.scoremap_dropout_prob,
                                        noise_shape=[1, 1, 21])
            scoremap *= self.scoremap_dropout_prob

        data_dict['scoremap'] = scoremap

        if self.scale_to_size:
            image, keypoint_uv21, keypoint_vis21 = data_dict['image'], data_dict['keypoint_uv21'], data_dict['keypoint_vis21']
            s = image.get_shape().as_list()
            image = tf.image.resize_images(image, self.scale_target_size)
            scale = (self.scale_target_size[0]/float(s[0]), self.scale_target_size[1]/float(s[1]))
            keypoint_uv21 = tf.stack([keypoint_uv21[:, 0] * scale[1],
                                      keypoint_uv21[:, 1] * scale[0]], 1)

            data_dict = dict()  # delete everything else because the scaling makes the data invalid anyway
            data_dict['image'] = image
            data_dict['keypoint_uv21'] = keypoint_uv21
            data_dict['keypoint_vis21'] = keypoint_vis21

        elif self.random_crop_to_size:
            tensor_stack = tf.concat([data_dict['image'],
                                      tf.expand_dims(tf.cast(data_dict['hand_parts'], tf.float32), -1),
                                      tf.cast(data_dict['hand_mask'], tf.float32)], 2)
            s = tensor_stack.get_shape().as_list()
            tensor_stack_cropped = tf.random_crop(tensor_stack,
                                                  [self.random_crop_size, self.random_crop_size, s[2]])
            data_dict = dict()  # delete everything else because the random cropping makes the data invalid anyway
            data_dict['image'], data_dict['hand_parts'], data_dict['hand_mask'] = tensor_stack_cropped[:, :, :3],\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 3], tf.int32),\
                                                                                  tf.cast(tensor_stack_cropped[:, :, 4:], tf.int32)
        sess = tf.Session()
        
        data_dict['image'] = image
        for key, value in data_dict.items():   
            value = sess.run(value)         
            # value = value_evaluated.numpy()
            if isinstance(value, np.ndarray):
                torch_value = torch.Tensor(value)
            else:
                # If value is a single value, put it into a list and convert it to torch.Tensor
                torch_value = torch.Tensor([value])
            if key == 'image':
                torch_value = torch_value.permute(2, 0, 1)
            data_dict[key] = torch.Tensor(torch_value)
            
        sess.close()

        data_dict['img_name'] = img_name
        names, tensors = zip(*data_dict.items())

        return dict(zip(names, tensors))



    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        # print('valid_vec.shape', valid_vec.shape)

        with tf.name_scope('create_multiple_gaussian_map'):
            sigma = tf.cast(sigma, tf.float32)
            assert len(output_size) == 2
            s = coords_uv.get_shape().as_list()
            coords_uv = tf.cast(coords_uv, tf.int32)
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
            cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
            cond_in = tf.logical_and(cond_1_in, cond_2_in)
            # print('cond_val.shape', cond_val.shape)
            # print('cond_in.shape', cond_in.shape)

            cond = tf.logical_and(cond_val, cond_in)

            coords_uv = tf.cast(coords_uv, tf.float32)

            # create meshgrid
            x_range = tf.expand_dims(tf.range(output_size[0]), 1)
            y_range = tf.expand_dims(tf.range(output_size[1]), 0)

            X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
            Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

            X.set_shape((output_size[0], output_size[1]))
            Y.set_shape((output_size[0], output_size[1]))

            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)

            X_b = tf.tile(X, [1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, s[0]])

            X_b -= coords_uv[:, 0]
            Y_b -= coords_uv[:, 1]

            dist = tf.square(X_b) + tf.square(Y_b)

            # print('dist.shape', dist.shape)
            # print('sigma.shape', sigma.shape)
            # print('cond.shape', cond.shape)
            scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

            return scoremap


if __name__ == '__main__':
    dataset_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'

    transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

    # Creating the dataset
    dataset = RHD_HandKeypointsDataset(root_dir=dataset_dir, set_type='evaluation', transform=transforms, debug=False)

    # Creating the DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
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
        print('keypoints_xyz', keypoints_xyz)
        print('')
        print('keypoints_xyz[:, :3]', keypoints_xyz[:, :3])
        print('keypoint_xyz21[:, :3]', keypoint_xyz21[:, :3])
        print('keypoint_uv21[:, :3]', keypoint_uv21[:, :3])
        print('keypoint_vis21[:, :3]', keypoint_vis21[:, :3])
        break

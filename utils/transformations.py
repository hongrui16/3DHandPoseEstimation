import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

from PIL import Image, ImageChops
import cv2
import torchvision.transforms.functional as TF
import sys

# sys.path.append('../')  
from config.config import *


'''
    RHD dataset
        sample = {'image': image, 'mask': mask, 'depth': depth,
                  'kp_coord_uv': kp_coord_uv, 'kp_visible': kp_visible,
                  'kp_coord_xyz': kp_coord_xyz, 'camera_matrix': camera_intrinsic_matrix}

'''

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel. RGB order
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        # img_name = sample['img_name']
        img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['mask'] = mask
        return sample


class Centeralize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel. RGB order
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['mask']
        # img_name = sample['img_name']
        img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['mask'] = mask
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['mask']
        # img_name = sample['img_name']
        
        img = np.array(img)
        if img.ndim == 3:
           img = img.astype(np.float32).transpose((2, 0, 1))
        else:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['mask'] = mask
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, args = None):
        self.args = args
    def __call__(self, sample):
        if self.args.distinguish_left_right_semantic:
            return sample
        img = sample['image']
        mask = sample['mask']
        if random.random() < 0.25:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['mask'] = mask
        return sample

class RandomVerticalFlip(object):
    def __init__(self, args = None):
        self.args = args
    def __call__(self, sample):
        if self.args.distinguish_left_right_semantic:
            return sample
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.25:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomRotate(object):
    def __init__(self, degree, args = None):
        self.degree = degree
        self.fill = args.ignore_index
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR, fillcolor = 0)
            mask = mask.rotate(rotate_degree, Image.NEAREST, fillcolor = self.fill)
            # return {'image': img,
            #         'label': mask}
            sample['image'] = img
            sample['label'] = mask
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.85:
            return sample
        img = sample['image']
        mask = sample['label']
        

        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.random()))

        sample['image'] = img
        sample['label'] = mask
        return sample
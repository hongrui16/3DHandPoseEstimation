import os
import numpy as np
import cv2
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch 

def plot_uv_on_image(keypoints_uv, image, keypoints_vis = None, horizon_flip = False, img_filepath = None):
    # print(f'keypoints_vis.shape: {keypoints_vis.shape}')
    # Adjust keypoints to ensure they are within the image boundaries of 320x320
    if not keypoints_vis is None:
        keypoints_uv = keypoints_uv[keypoints_vis]
    
    if horizon_flip:
        image = image[:,::-1]
        h, w, _ = image.shape
        keypoints_uv[:, 0] = w - keypoints_uv[:, 0]

    # Re-plot the keypoints on the image, this time ensuring they are within the image boundaries
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.scatter(keypoints_uv[:, 0], keypoints_uv[:, 1], c='red', s=10)  # plot keypoints in red
    plt.axis('on')  # remove axes for better visualization
    
    if not img_filepath is None:
        plt.savefig(img_filepath)
    else:
        plt.show()


def plot_mask_on_image(hand_map_l, hand_map_r, image, img_filepath = None):
    # Convert binary masks to RGB color masks
    hand_mask_l_color = np.zeros((hand_map_l.shape[0], hand_map_l.shape[1], 3), dtype=np.uint8)
    hand_mask_r_color = np.zeros((hand_map_r.shape[0], hand_map_r.shape[1], 3), dtype=np.uint8)

    hand_mask_l_color[hand_map_l.numpy() == 1] = [255, 0, 0]  # Red for left hand
    hand_mask_r_color[hand_map_r.numpy() == 1] = [0, 255, 0]  # Green for right hand

    # Overlay color masks on the original image
    overlay_image = image.copy()
    overlay_image[hand_mask_l_color[:, :, 0] == 255] = hand_mask_l_color[hand_mask_l_color[:, :, 0] == 255]
    overlay_image[hand_mask_r_color[:, :, 1] == 255] = hand_mask_r_color[hand_mask_r_color[:, :, 1] == 255]

    # Display the result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay_image)
    plt.axis('on')

    # Calculate centroids and add text labels for left and right hand masks
    l_y, l_x = np.where(hand_map_l.numpy() == 1)
    if len(l_x) > 0 and len(l_y) > 0:  # Check if left hand mask exists
        plt.text(l_x.mean(), l_y.mean(), 'Left', color='white', fontsize=12, ha='center', va='center')

    r_y, r_x = np.where(hand_map_r.numpy() == 1)
    if len(r_x) > 0 and len(r_y) > 0:  # Check if right hand mask exists
        plt.text(r_x.mean(), r_y.mean(), 'Right', color='white', fontsize=12, ha='center', va='center')

    if not img_filepath is None:
        plt.savefig(img_filepath)
    else:
        plt.show()


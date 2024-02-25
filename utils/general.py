import torch
import torch.nn.functional as F
import numpy as np


def crop_image_from_xy_torch(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image in PyTorch.

    Inputs:
        image: 3D tensor, [channels, height, width]
        crop_location: tensor, [2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 3D tensor, [channels, crop_size, crop_size]
    """
    assert len(image.shape) == 3, "Image needs to be of shape [channels, height, width]"

    channels, height, width = image.shape
    # print(f'image.shape: {image.shape}, crop_location.shape: {crop_location.shape}')
    # print(f'crop_location: {crop_location}, crop_size: {crop_size}, scale: {scale}')
    # Calculate scaled crop size
    crop_size_scaled = int(crop_size / scale)

    # Initialize an empty tensor for cropped images
    # cropped_images = torch.empty((image.shape[0], image.shape[1], crop_size_scaled, crop_size_scaled))
    # print('cropped_images.shape', cropped_images.shape)
    # Calculate crop coordinates
    y1 = int(crop_location[0] - crop_size_scaled // 2) if int(crop_location[0] - crop_size_scaled // 2) > 0 else 0
    y2 = y1 + crop_size_scaled if y1 + crop_size_scaled < height else height

    x1 = int(crop_location[1] - crop_size_scaled // 2) if int(crop_location[1] - crop_size_scaled // 2) > 0 else 0
    x2 = x1 + crop_size_scaled if x1 + crop_size_scaled < width else width
    # print(f'y1 {y1} - y2 {y2}; x1 {x1} - x2 {x2}')
    # Crop and resize
    cropped_img = image[:, y1:y2, x1:x2]
    # print('cropped_img.shape', cropped_img.shape)
    cropped_img = F.interpolate(cropped_img, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
    # print('cropped_img.shape', cropped_img.shape)


    return cropped_img




def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90., elev=90.)




def calculate_padding(input_size, kernel_size, stride):
    """Calculates the amount of padding to add according to Tensorflow's
    padding strategy."""

    cond = input_size % stride

    if cond == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - cond, 0)

    if pad % 2 == 0:
        pad_val = pad // 2
        padding = (pad_val, pad_val)
    else:
        pad_val_start = pad // 2
        pad_val_end = pad - pad_val_start
        padding = (pad_val_start, pad_val_end)

    return padding


def _get_rot_mat(ux_b, uy_b, uz_b):
    """Converts axis-angle parameters to a rotation matrix.

    The axis-angle parameters have an encoded angle.

    Args:
        ux, uy, uz axis-angle parameters， Tensor (batch x 1):

    Returns:
        rot_matrix - Tensor (batch x 3 x 3): Rotation matrices.
    """
    """Returns a rotation matrix from axis and (encoded) angle in PyTorch."""
    u_norm = torch.sqrt(ux_b**2 + uy_b**2 + uz_b**2 + 1e-8)
    theta = u_norm

    # some tmp vars
    st_b = torch.sin(theta)
    ct_b = torch.cos(theta)
    one_ct_b = 1.0 - torch.cos(theta)

    st = st_b[:, 0]
    ct = ct_b[:, 0]
    one_ct = one_ct_b[:, 0]
    norm_fac = 1.0 / u_norm[:, 0]

    ux = ux_b[:, 0] * norm_fac
    uy = uy_b[:, 0] * norm_fac
    uz = uz_b[:, 0] * norm_fac

    top = torch.stack((ct + ux * ux * one_ct, ux * uy * one_ct - uz * st, ux * uz * one_ct + uy * st), dim=1)
    mid = torch.stack((uy * ux * one_ct + uz * st, ct + uy * uy * one_ct, uy * uz * one_ct - ux * st), dim=1)
    bot = torch.stack((uz * ux * one_ct - uy * st, uz * uy * one_ct + ux * st, ct + uz * uz * one_ct), dim=1)

    rot_matrix = torch.stack((top, mid, bot), dim=1)

    return rot_matrix

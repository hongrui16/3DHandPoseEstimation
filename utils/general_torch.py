import torch
import torch.nn.functional as F

def crop_image_from_xy_torch(image, crop_location, crop_size, scale=1.0):
    """
    Crops an image in PyTorch.

    Inputs:
        image: 4D tensor, [batch, channels, height, width]
        crop_location: tensor, [batch, 2] which represent the height and width location of the crop
        crop_size: int, describes the extension of the crop
    Outputs:
        image_crop: 4D tensor, [batch, channels, crop_size, crop_size]
    """
    assert len(image.shape) == 4, "Image needs to be of shape [batch, channels, height, width]"

    # Calculate scaled crop size
    crop_size_scaled = int(crop_size / scale)

    # Initialize an empty tensor for cropped images
    cropped_images = torch.empty((image.shape[0], image.shape[1], crop_size_scaled, crop_size_scaled))

    for i in range(image.shape[0]):
        # Calculate crop coordinates
        y1 = int(crop_location[i, 0] - crop_size_scaled // 2)
        y2 = y1 + crop_size_scaled
        x1 = int(crop_location[i, 1] - crop_size_scaled // 2)
        x2 = x1 + crop_size_scaled

        # Crop and resize
        cropped_img = image[i, :, y1:y2, x1:x2]
        cropped_img = F.interpolate(cropped_img.unsqueeze(0), size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        cropped_images[i] = cropped_img.squeeze(0)

    return cropped_images
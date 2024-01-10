import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


from config.config import *

# from network.sub_modules.conditionalDiffusion import *
# from network.sub_modules.diffusionJointEstimation import DiffusionJointEstimation
# from network.sub_modules.resNetFeatureExtractor import ResNetFeatureExtractor
# from network.sub_modules.forwardKinematicsLayer import ForwardKinematics

from network.diffusion3DHandPoseEstimation import Diffusion3DHandPoseEstimation
from dataloader.dataloaderRHD import RHD_HandKeypointsDataset
from criterions.loss import LossCalculation

flag = torch.cuda.is_available()
if flag:
    print("CUDA is available")
    device = "cuda"
else:
    print("CUDA is unavailable")
    device = "cpu"


model = Diffusion3DHandPoseEstimation(device=device)

criterion = LossCalculation(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_set = RHD_HandKeypointsDataset(root_dir=dataset_root_dir, set_type='training')
val_set = RHD_HandKeypointsDataset(root_dir=dataset_root_dir, set_type='evaluation')


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


def trainval(cur_epoch, total_epoch, model, loader, criterion, optimizer, split):
    assert split in ['training', 'validation']
    if split == 'training':
        model.train()
    else:
        model.eval()
    tbar = tqdm(loader)
    num_iter = len(loader)

    width = 10  # Total width including the string length
    formatted_split = split.rjust(width)

    # if self.args.master:
    #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
    for iter, sample in enumerate(tbar):
        keypoint_uv21 = sample['keypoint_uv21'].to(device)
        keypoint_xyz21_normed = sample['keypoint_xyz21_normed'].to(device)
        keypoint_vis21 = sample['keypoint_vis21'].to(device)
        image = sample['image'].to(device)
        cam_mat = sample['cam_mat'].to(device)
        index_root_bone_length = sample['keypoint_scale'].to(device)

        optimizer.zero_grad()

        refined_joint_coord, diffusion_loss, resnet_features = model(image, cam_mat, keypoint_xyz21_normed)
        pred_xyz_coordinates, pred_uv_coordinates = refined_joint_coord

        loss_part1 = criterion(pred_xyz_coordinates, keypoint_xyz21_normed, pred_uv_coordinates, keypoint_uv21, keypoint_vis21) 
        loss = loss_part1 + diffusion_loss
        if split == 'training':
            loss.backward()
            optimizer.step()

        if iter % 10 == 0:
            tbar.set_description(f'{formatted_split} Epoch: {cur_epoch:03d}/{total_epoch:03d}, Epoch: {iter:05d}/{num_iter:05d} loss: {loss:.5f}')



for epoch in range(0, max_epoch): 
    trainval(epoch, max_epoch, model, train_loader, criterion, optimizer, 'training')
    trainval(epoch, max_epoch, model, val_loader, criterion, optimizer, 'validation')

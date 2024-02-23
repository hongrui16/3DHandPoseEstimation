# DenoisingDiffusion3DHandPoseEstimation
This a code implementation for 3D hand pose estimation, which contains a bunch of networks.
<!-- 
## Install library packages
```pip install -r requirements.txt``` -->


## 1 Parameter setting
All parameters are configured in ```config/config.py```. This includes settings such as batch size, the input channel of the network, etc.

## 2 Dataset
Currently, only a data loader for RHD dataset is impletemented.
To use the RHD dataset, specify the data directory and the dataset name ```dataset_root_dir``` and ```dataset_name``` in ```config/config.py```, respectively.
### 2.1 RHD
Joint order in RHD is as follows:

<img src="imgs/RHD_Joint_Order.png" width="400" height="400">

## 3 Network
The network comprises various elementary modules, including conditional diffusion, a forward kinematic layer, and a ResNet feature extractor, etc.
They are located in ```network/sub_modules```.
The global network architecture is impletemented in ```network/diffusion3DHandPoseEstimation.py```.


## 4 Loss function
All loss functions are implemented in ```criterions/loss.py```. The loss function specific to the diffusion model is included in its module ```network/sub_modules/conditionalDiffusion.py```. The computation considers only the visible points.

## 5 Metrics (MPJPE)
The metric MPJPE (Mean Per Joint Position Error) is implemented in  ```criterions/metric.py```, and it also accounts for only the visible points.

## 6 Training and Validation
Training and validation processes are implemented in ```wroker.py```. For debugging purposes, you can set the input variable ```fast_debug``` of the ```trainval function``` to ```True```.

## 7 MANO
Joint order in MANO is as follows:

<img src="imgs/MANO_Joint_Order.png" width="300" height="300">

## 8 Relative References
Learning to Estimate 3D Hand Pose from Single RGB Images [official code](https://github.com/lmb-freiburg/hand3d); [pytorch code](https://github.com/ajdillhoff/colorhandpose3d-pytorch)

[Learning Joint Reconstruction of Hands and Manipulated Objects](https://github.com/hassony2/obman_train/tree/master)

[3D Hand Shape and Pose from Images in the Wild](https://github.com/boukhayma/3dhand)

[manopth](https://github.com/hassony2/manopth)

[MANO](https://github.com/otaheri/MANO)


[3D Hand Pose Estimation from Single RGB Camera](https://github.com/OlgaChernytska/3D-Hand-Pose-Estimation)

[Contrastive Representation Learning for Hand Shape Estimation](https://github.com/lmb-freiburg/contra-hand/tree/main)

["Denoising Diffusion for 3D Hand Pose Estimation from Images"](https://arxiv.org/abs/2308.09523)

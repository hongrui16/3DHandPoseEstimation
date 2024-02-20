'''## dataset parameters'''
dataset_root_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'
dataset_name = 'RHD'

'''## dataloader parameters'''
shuffle=True
use_wrist_coord=True
sigma=25.0
hand_crop=True ## this must be True
random_crop_to_size=False
scale_to_size=False
hue_aug=False
coord_uv_noise=False
crop_center_noise=False
crop_scale_noise=False
crop_offset_noise=False
scoremap_dropout=False
calculate_scoremap=False

'''## training parameters'''
save_log_dir = 'logs'
max_epoch = 100
finetune = False
batch_size = 480
gpu_idx = None
# gpu_idx = 1
uv_from_xD = 3 ## this is for where to get output of UV coordinates in TwoDimHandPoseWithFKEstimation, x = 2: from 2D keypoints directly; x = 3: from 3D keypoints projections; x = 2.5: 1/2 from 2D keypoints directly, 1/2 from 3D keypoints projections
is_inference = False

'''model parameters''' 
# model_name = 'TwoDimHandPose'
# model_name = 'DiffusionHandPose'
# model_name = 'ThreeDimHandPose'
# model_name = 'OnlyThreeDimHandPose'
model_name = 'TwoDimHandPoseWithFK'


'''## inference parameters'''
infer_batch_size = 100
# resume_weight_path = 'logs/TwoDimHandPose/RHD/run_000/DF_model_best.pth.tar'
# resume_weight_path = 'logs/ThreeDimHandPose/RHD/run_000/DF_model_best.pth.tar'
# resume_weight_path = 'logs/OnlyThreeDimHandPose/RHD/run_000/DF_model_best.pth.tar'
resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-15-52/DF_model_best.pth.tar'
resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-16-45/DF_model_best.pth.tar'
resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-17-45/DF_model_best.pth.tar'


'''model parameters''' 
# model_name = 'TwoDimHandPose'
# model_name = 'DiffusionHandPose'
# model_name = 'ThreeDimHandPose'
# model_name = 'OnlyThreeDimHandPose'
model_name = 'TwoDimHandPoseWithFK'


'''## general parameters'''
keypoint_num = 21


'''## diffusion3DHandPoseEstimation parameters '''
condition_feat_dim = 256
num_timesteps = 400
num_sampling_timesteps = 200
keypoint_feat_Ch = 1
bone_length_num = 20
other_joint_angles_num = 23

## twoDimHandPoseEstimation parameters
resnet_out_feature_dim = 1024

'''## MANO parameters'''
mano_right_hand_path = '../config/mano/models/MANO_RIGHT.pkl'
mano_pose_num = 45
mano_beta_num = 10 ### do not change this
import platform
'''## dataset parameters'''
if platform.system() == 'Windows':
    dataset_root_dir = 'dataset/RHD'
elif platform.system() == 'Linux':
    dataset_root_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'

dataset_name = 'RHD'

'''## dataloader parameters'''
shuffle=True
num_workers = 10

use_wrist_coord=False # Ture: use wrist coordinate; False: use palm center coordinate
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
calculate_scoremap=True ## this must be True
use_val_dataset_to_debug = True


'''network parameters''' 
# model_name = 'TwoDimHandPose'
model_name = 'DiffusionHandPose'
# model_name = 'ThreeDimHandPose'
# model_name = 'OnlyThreeDimHandPose'
# model_name = 'TwoDimHandPoseWithFK'
# model_name = 'MANO3DHandPose'
# model_name = 'ThreeHandShapeAndPoseMANO'
# model_name = 'Resnet50MANO3DHandPose'

# model_name = 'Hand3DPoseNet'
# model_name = 'Hand3DPosePriorNetwork'

input_channels = 24 #3, 21, 24



'''## general parameters'''
keypoint_num = 21
gpu_idx = None
# gpu_idx = 2
resnet_out_feature_dim = 1024
compute_uv_loss = False

'''## diffusion3DHandPoseEstimation parameters '''
condition_feat_dim = 256
num_timesteps = 400
num_sampling_timesteps = 200
keypoint_feat_Ch = 1
bone_length_num = 20
other_joint_angles_num = 23


'''## MANO parameters'''
mano_right_hand_path = 'config/mano/models/MANO_RIGHT.pkl'
mano_pose_num = 10 #6, 10, 45
mano_beta_num = 10 ### do not change this
joint_order_switched = True

'''## ThreeHandShapeAndPose network ##'''
network_regress_uv = False



'''## training parameters'''
save_log_dir = 'logs'
max_epoch = 60
finetune = False
batch_size = 480
# batch_size = 1
uv_from_xD = 3 ## this is for where to get output of UV coordinates in TwoDimHandPoseWithFKEstimation, x = 2: from 2D keypoints directly; x = 3: from 3D keypoints projections; x = 2.5: 1/2 from 2D keypoints directly, 1/2 from 3D keypoints projections
is_inference = False
resume_weight_path = None
# resume_weight_path = 'logs/Hand3DPoseNet/RHD/run_2024-02-22-03-08-39/DF_model_best.pth.tar'
use_fake_data = True # True: use fake data for debug; False: use real data


'''## inference parameters'''
infer_batch_size = 100
# infer_resume_weight_path = 'logs/TwoDimHandPose/RHD/run_000/DF_model_best.pth.tar'
# infer_resume_weight_path = 'logs/ThreeDimHandPose/RHD/run_000/DF_model_best.pth.tar'
# infer_resume_weight_path = 'logs/OnlyThreeDimHandPose/RHD/run_000/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-15-52/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-16-45/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/TwoDimHandPoseWithFK/RHD/run_2024-02-15-00-17-45/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/MANO3DHandPose/RHD/run_2024-02-20-20-45-23/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/MANO3DHandPose/RHD/run_2024-02-20-20-45-23/DF_checkpoint.pth.tar'
infer_resume_weight_path = 'logs/MANO3DHandPose/RHD/run_2024-02-20-21-23-46/DF_checkpoint.pth.tar'
infer_resume_weight_path = 'logs/MANO3DHandPose/RHD/run_2024-02-22-09-31-39/DF_checkpoint.pth.tar'
infer_resume_weight_path = 'logs/Hand3DPosePriorNetwork/RHD/run_2024-02-29-17-16-45/DF_model_best.pth.tar'
infer_resume_weight_path = 'logs/Hand3DPosePriorNetwork/RHD/run_2024-02-29-17-16-45/DF_checkpoint.pth.tar'




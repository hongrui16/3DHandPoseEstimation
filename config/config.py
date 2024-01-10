dataset_root_dir = 'dataset/RHD'
save_log_dir = 'logs'
dataset_name = 'RHD'

max_epoch = 100


condition_feat_dim = 256
num_timesteps = 500
num_sampling_timesteps = 100
keypoint_feat_Ch = 1
keypoint_num = 21
batch_size = 32
bone_length_num = 20
other_joint_angles_num = 23

shuffle=True
use_wrist_coord=True
sigma=25.0
hand_crop=False
random_crop_to_size=False
scale_to_size=False
hue_aug=False
coord_uv_noise=False
crop_center_noise=False
crop_scale_noise=False
crop_offset_noise=False
scoremap_dropout=False
dataset_root_dir = '/home/rhong5/research_pro/hand_modeling_pro/dataset/RHD/RHD'
save_log_dir = 'logs'
dataset_name = 'RHD'

max_epoch = 100

# resume_weight_path = 'logs/RHD/run_000/DF_model_best.pth.tar'
resume_weight_path = None
finetune = False


condition_feat_dim = 256
num_timesteps = 400
num_sampling_timesteps = 200
keypoint_feat_Ch = 1
keypoint_num = 21
batch_size = 480
bone_length_num = 20
other_joint_angles_num = 23

shuffle=True
use_wrist_coord=True
sigma=25.0
hand_crop=True
random_crop_to_size=False
scale_to_size=False
hue_aug=False
coord_uv_noise=False
crop_center_noise=False
crop_scale_noise=False
crop_offset_noise=False
scoremap_dropout=False
calculate_scoremap=False
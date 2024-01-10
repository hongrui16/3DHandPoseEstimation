import torch
# from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
# from DenoisingDiffusion3DHandPoseEstimation.utils.DenoiseDiffusion import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import Adam
from ema_pytorch import EMA
import sys


from network.sub_modules.conditionalDiffusion import *


def demo():

    sample_num = 64
    condition_feat_dim =  512
    batch_size = 4
    seq_length = 63
    channel = 1
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = channel,
        condition_feat_dim = condition_feat_dim
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_length,
        timesteps = 500,
        sampling_timesteps = 200,
        objective = 'pred_v'
    )

    x0 = torch.rand(sample_num, channel, seq_length) # features are normalized from 0 to 1

    dataset = Dataset1D(x0)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

    flag = torch.cuda.is_available()
    if flag:
        print("CUDA is available")
        device = "cuda"
    else:
        print("CUDA is unavailable")
        device = "cpu"

    def simplyTrain():
        condition_feat = torch.rand(sample_num, condition_feat_dim)

        loss = diffusion(x0, condition_feat)
        loss.backward()
        print('loss', loss)


    def train():
        train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True)
        train_pbar = tqdm(train_dataloader)
        ######################## train ######################################################
        # diffusion = diffusion.to(device)

        model.train()

        for i, x in enumerate(train_pbar):
            # print('images', images.shape)
            # x = x.to(device)
            # print(f'x.shape', x.shape) # torch.Size([4, 32, 128])
            condition_feat = torch.rand(batch_size, condition_feat_dim)

            loss = diffusion(x, condition_feat)
            loss.backward()
            # print('loss', loss)
            break
        print('train end........')



    def sample():
        # after a lot of training
        sampled_seq = diffusion.sample(batch_size = 4)
        sampled_seq.shape # (4, 32, 128)


    simplyTrain()
    train()
    # trainer()

def demo2():
            
    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 32
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 128,
        timesteps = 1000,
        objective = 'pred_v'
    )

    training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1
    dataset = Dataset1D(training_seq)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

    loss = diffusion(training_seq)
    loss.backward()
    print(f'loss: {loss}')


# demo2()
demo()
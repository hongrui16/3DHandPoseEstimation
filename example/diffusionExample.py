import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
# from utils import *
# from modules import UNet
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os, sys
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter
import glob
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from PIL import Image

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # print(f'    x: {x.shape}, emb: {emb.shape}')
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print(f'forward    x: {x.shape}, t: {t.shape}')
        x1 = self.inc(x)
        # print(f'forward    x1: {x1.shape}, t: {t.shape}')
        x2 = self.down1(x1, t)
        # print(f'forward    x2: {x2.shape}, t: {t.shape}')
        # x2 = self.sa1(x2)
        # print(f'forward    x2: {x2.shape}, t: {t.shape}')
        x3 = self.down2(x2, t)
        # x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        # x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        # x = self.sa4(x)
        x = self.up2(x, x2, t)
        # x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output




def save_images(images, path, **kwargs):
    print(f'save to {path}\n')
    # logging.info(f'save to {path}\n')
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # im = Image.fromarray(ndarr)
    # im.save(path)
    cv2.imwrite(path, ndarr)
    return ndarr


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()  # 切换模型到评估模式，这通常会关闭诸如dropout等训练特有的层
        with torch.no_grad():  # 禁用梯度计算，这在推理过程中节省计算资源
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)  # 初始化一个随机噪声图像批量
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):  # 使用 tqdm 逆向遍历噪声步骤，显示进度条
                t = (torch.ones(n) * i).long().to(self.device)  # 创建一个常数张量，表示当前时间步骤
                predicted_noise = model(x, t)  # 使用模型预测当前时间步骤的噪声
                alpha = self.alpha[t][:, None, None, None]  # 获取当前步骤的alpha值，并调整形状以匹配数据批量的形状
                alpha_hat = self.alpha_hat[t][:, None, None, None]  # 获取当前步骤的alpha_hat值，并进行形状调整
                beta = self.beta[t][:, None, None, None]  # 获取当前步骤的beta值，并进行形状调整
                if i > 1:  # 如果不是最后一个步骤
                    noise = torch.randn_like(x)  # 生成随机噪声
                else:  # 如果是最后一个步骤
                    noise = torch.zeros_like(x)  # 使用零噪声

                # 更新 x 使用预测的噪声和计算出的 alpha, alpha_hat, beta 值
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                '''
                + torch.sqrt(beta) * noise: 在逆过程中，即使我们从图像中去除了预测的噪声，仍然需要向图像中添加一些随机噪声 noise。
                这是因为扩散模型不是简单地撤销噪声，而是通过预测噪声并添加新的随机噪声来进行迭代恢复。
                在非最后一步（i > 1），添加的是随机噪声，这帮助模型探索可能的原始图像变化，并防止过拟合到预测的噪声。
                在最后一步（i == 1），添加的是零噪声（即没有额外噪声），因为此时我们希望得到尽可能准确的原始图像重建。
                '''
        model.train()  # 将模型切换回训练模式
        x = (x.clamp(-1, 1) + 1) / 2  # 将图像数据的值规范化到[0, 1]范围
        x = (x * 255).type(torch.uint8)  # 将数据范围从[0, 1]转换到[0, 255]并转换为无符号8位整型
        return x  # 返回生成的图像



def get_dataset(dataset_name, batch_size = 128):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)


        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

        trainloader = DataLoader(trainset, batch_size=batch_size, drop_last=True, 
                                                shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, drop_last=True,
                                                shuffle=False, num_workers=2)
        return trainloader, testloader
    else:
        None


def save_checkpoint(state, is_best, model_name='', ouput_weight_dir = ''):
    """Saves checkpoint to disk"""
    os.makedirs(ouput_weight_dir, exist_ok=True)
    best_model_filepath = os.path.join(ouput_weight_dir, f'{model_name}_model_best.pth.tar')
    filename = os.path.join(ouput_weight_dir, f'{model_name}_checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:   
        torch.save(state, best_model_filepath)



def plot_loss(trainingEpoch_loss, valEpoch_loss, exp_dir):
    # fig = plt.figure()
    plt.plot(trainingEpoch_loss, label='train_epoch_loss')
    plt.plot(valEpoch_loss, label='val_epoch_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(exp_dir, f"loss.jpg"))
    plt.legend()
    plt.show()
    # plt.close()

def plot_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    


def calculate_fid(diffusion, model):
    # example of calculating the frechet inception distance in Keras for cifar10

    # scale an array of images to a new size
    def scale_images(images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)

    # calculate frechet inception distance
    def calculate_fid(model, images1, images2):
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # prepare the inception v3 model
    model_inv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # load cifar10 images
    # (images1, _), (images2, _) = cifar10.load_data()
    images1 = diffusion.sample(model,256)
    images2 = diffusion.sample(model,256)
    images1 = images1.permute(0, 2, 3, 1).to('cpu').numpy()
    images2 = images2.permute(0, 2, 3, 1).to('cpu').numpy()
    print('Loaded', 'images1.shape', images1.shape, 'images2.shape', images2.shape)
    # print('images1', type(images1))
    # plt.figure(figsize=(8,3)) 
    #     # plt.figure() 
    # for i in range(5):
    # 	# print(img)
    # 	# print(len(img))
    # 	# print('img[0].shape', img[0].shape)
    # 	img1 = images1[i].copy()
    # 	# img1 = img1.transpose(2, 0, 1)
    # 	plt.subplot(2, 5, i + 1)
    # 	plt.imshow(img1)

    # 	img2 = images2[i].copy()
    # 	# img2 = img2.transpose(2, 0, 1)
    # 	plt.subplot(2, 5, i + 6)
    # 	plt.imshow(img2)
    # 	# print('img', img.min(), img.max(), img.shape)

    shuffle(images1)
    images1 = images1[:10000]

    # images1 = images1[:10]
    # images2 = images2[:10]

    # # convert integer to floating point values
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (299,299,3))
    images2 = scale_images(images2, (299,299,3))
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate fid
    fid = calculate_fid(model_inv3, images1, images2)
    print('FID: %.3f' % fid)


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on the GPU")
    else:
        device = torch.device("cpu")
        print("running on the CPU")


    # setup_logging(args.run_name)
    log_dir = sorted(glob.glob(os.path.join('logs', args.run_name, 'run_*')), key=lambda x: int(x.split('_')[-1]))
    run_id = int(log_dir[-1].split('_')[-1]) + 1 if log_dir else 0
    exp_dir = os.path.join('logs', args.run_name, 'run_{}'.format(str(run_id)))
    os.makedirs(exp_dir, exist_ok=True)
    print('exp_dir', exp_dir)

    output_img_dir = os.path.join(exp_dir, 'result')
    os.makedirs(output_img_dir, exist_ok=True)

    # device = args.device
    # dataloader = get_data(args)
    train_dataloader, val_dataloader = get_dataset('CIFAR10', batch_size = args.batch_size)
    
    if args.run_name == 'UNet':
        model = UNet().to(device)
    # elif args.run_name == 'SimpleUNet':
    #     model = SimpleUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)


    logger = SummaryWriter(exp_dir)

    min_val_loss = float('inf')
    
    epoch_step = 2
    generate_image_num = 32

    start_epoch = 0
    trainingEpoch_loss = []
    valEpoch_loss = []
    weight_filepath = args.resume
    fine_tune = False
    if weight_filepath and os.path.exists(weight_filepath):
        checkpoint = torch.load(weight_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        if not fine_tune:
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            trainingEpoch_loss += checkpoint['trainingEpoch_loss']
            valEpoch_loss += checkpoint['valEpoch_loss']

    logging.info(f"Starting epoch {start_epoch}:\n")
    for epoch in range(start_epoch, args.epochs):
        
        train_pbar = tqdm(train_dataloader)
        train_step_loss = []
        ######################## train ######################################################
        model.train()
        split = 'train'
        for i, (images, _) in enumerate(train_pbar):
            # print('images', images.shape)
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # print(f'train t.shape: {t.shape}, t: {t}') # t.shape: torch.Size([batch_size]), t: tensor([302, 668], device='cuda:0'
            x_t, noise = diffusion.noise_images(images, t)
            # print(f'train x_t.shape: {x_t.shape}, noise.shape: {noise.shape}') # x_t.shape: torch.Size([2, 3, 32, 32]), noise.shape: torch.Size([2, 3, 32, 32])

            predicted_noise = model(x_t, t)
            # print(f'train predicted_noise.shape: {predicted_noise.shape}') # torch.Size([2, 3, 32, 32])

            loss = mse(noise, predicted_noise)
            # print(f'train loss.shape: {loss.shape}') # loss.shape: torch.Size([])


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.set_postfix()
            MSE=loss.item()
            
            # pbar.set_description(f"Epoch {epoch} {split.ljust(6)} | iter {i:04d} Loss: {MSE:4f}")
            train_pbar.set_description(f"Epoch {epoch} train | iter {i:04d} Loss: {MSE:4f} ")
            # test_tbar.set_description(f"Epoch {epoch} 'test'  | iter {iter:03d} Loss: {loss.item():4f} ")
            train_step_loss.append(round(MSE, 4))

        # if epoch % epoch_step == 0:
        #     # print('----------1--------here------------------------------\n')
        #     sampled_images = diffusion.sample(model, n=generate_image_num)        
        #     img = save_images(sampled_images, os.path.join(output_img_dir, f"train_{epoch}.jpg"))
        #     plot_img(img)
        trainingEpoch_loss.append(np.array(train_step_loss).mean())
        logger.add_scalar("Train Epoch MSE", loss.item(), global_step=epoch)
        
        # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

        ######################## val ######################################################
        model.eval()
        split = 'val'
        val_pbar = tqdm(val_dataloader)
        val_step_loss = []
        for i, (images, _) in enumerate(val_pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            with torch.no_grad():
                predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            MSE=loss.item()
            
            # val_pbar.set_description(f"Epoch {epoch} {split.ljust(6)} | iter {i:04d} Loss: {MSE:4f}")
            val_pbar.set_description(f"Epoch {epoch} val   | iter {i:04d} Loss: {MSE:4f} ")
            val_step_loss.append(round(MSE, 4))
        # print('----------2--------here------------------------------')
        if epoch % epoch_step == 0:
            sampled_images = diffusion.sample(model, n=generate_image_num)
            img = save_images(sampled_images, os.path.join(output_img_dir, f"val_{epoch}.jpg"))
            # plot_img(img)

        valEpoch_loss.append(np.array(val_step_loss).mean())
        logger.add_scalar("Val Epoch MSE", loss.item(), global_step=epoch)
        
        # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trainingEpoch_loss': trainingEpoch_loss,
                        'valEpoch_loss': valEpoch_loss,
            
                    }
        if np.array(val_step_loss).mean() < min_val_loss:
            min_val_loss = np.array(val_step_loss).mean()
            is_best = True
        else:
            is_best = False

        save_checkpoint(checkpoint, is_best, args.run_name, exp_dir)

    
    best_weight_filepath = os.path.join(exp_dir, f'{args.run_name}_model_best.pth.tar')
    if os.path.exists(best_weight_filepath):
        checkpoint = torch.load(best_weight_filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    sampled_images = diffusion.sample(model, n=generate_image_num)
    img = save_images(sampled_images, os.path.join(output_img_dir, f"{args.run_name}_generated.jpg"))
    plot_img(img)

    calculate_fid(diffusion, model)
    plot_loss(trainingEpoch_loss, valEpoch_loss, exp_dir)



def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "UNet"
    # args.run_name = "SimpleUNet"
    args.epochs = 50
    args.batch_size = 2
    args.image_size = 32
    # args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    # args.device = "cuda"
    args.lr = 3e-4
    args.resume = None
    # args.resume = 'D:/hongRui/GMU_course/CS782/Assigment2/Diffusion-Models-pytorch/logs/UNet/run_0/UNet_checkpoint.pth.tar'
    flag = torch.cuda.is_available()
    if flag:
        print("CUDA is available")
        args.device = "cuda"
    else:
        print("CUDA is unavailable")
        args.device = "cpu"


    main(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()

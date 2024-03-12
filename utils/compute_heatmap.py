import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import sys, os

sys.path.append('../')

from config import config as cfg

def render_gaussian_heatmap(joint_coord):
    x = torch.arange(cfg.output_hm_shape[2])
    y = torch.arange(cfg.output_hm_shape[1])
    z = torch.arange(cfg.output_hm_shape[0])
    zz,yy,xx = torch.meshgrid(z,y,x)
    xx = xx[None,None,:,:,:].float()
    yy = yy[None,None,:,:,:].float()
    zz = zz[None,None,:,:,:].float()
    
    x = joint_coord[:,:,0,None,None,None]
    y = joint_coord[:,:,1,None,None,None]
    z = joint_coord[:,:,2,None,None,None]
    heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
    heatmap = heatmap * 255
    return heatmap
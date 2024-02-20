import torch
from torchvision import models 
import sys
# sys.path.append('../')  


from config import config

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, feat_dim):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        self.fc = torch.nn.Linear(resnet.fc.in_features, feat_dim)  # Decrease feature dimensionality
    
    def forward(self, x):
        x = self.resnet_backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

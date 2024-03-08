import torch
from torchvision import models 
from torch import nn
import sys
# sys.path.append('../')  


from config import config

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, feat_dim):
        super(ResNetFeatureExtractor, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        # Modify conv1 for 21 input channels and change kernel size to 3
        self.feature_extractor.conv1 = nn.Conv2d(config.input_channels, self.feature_extractor.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Utilize the original fully connected layer, get the output channels
        self.num_output_features = self.feature_extractor.fc.out_features

        self.fc = torch.nn.Linear(self.num_output_features, feat_dim)  # Decrease feature dimensionality
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo

class Img_embed(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Img_embed, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self,x):
        x = x.view(1, -1)
        output = self.fc1(x)
        output = self.relu(output)
        output = F.normalize(self.fc2(output), dim=0, p=2)

        return output

class IMGNet(nn.Module):
    def __init__(self):
        self.model = Img_embed(256**2, 64**2, 300)
        self.model = torch.load("../models/img_model_best.pth")['img_model']

        for param in self.model.parameters():
        	param.resquires_grad = False
        print('Load pretrained img_embed model succeed')
    
    def forward(self, x):
        return self.model(x)

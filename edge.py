from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim

import os

output_rootdir= 'data/birds/CUB_200_2011/graysclae_image'
input_rootdir = 'data/birds/CUB_200_2011/images'
cnt = 0
for subfolder in os.listdir(input_rootdir):
    if not os.path.exists(output_rootdir+ '/' + subfolder):
        os.mkdir(output_rootdir+ '/' + subfolder)
    img_ls = os.listdir(input_rootdir+'/' + subfolder)
    for img_name in img_ls:
        img = cv2.imread(input_rootdir+ '/' + subfolder +'/' +img_name,0)
        edges = cv2.Canny(img,200,250)
        filename =output_rootdir+ '/' + subfolder + '/' + img_name
        cv2.imwrite(filename, edges)
        cnt += 1
    print(output_rootdir+ '/' + subfolder , '***', cnt)
    cnt = 0

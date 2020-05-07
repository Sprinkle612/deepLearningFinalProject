#!/usr/bin/env python
# coding: utf-8

# In[23]:


from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import pickle
import os
import re


# In[28]:


with open ("colors.pkl",'rb') as f:
    colors = pickle.load(f)


# In[37]:


colors.add('orange')
colors.add('grey')
colors.add('dark')
colors.add('bright')
colors.add('brown')


# In[39]:


colors_re = re.compile('|'.join(map(re.escape, colors)))


# In[40]:


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])


# In[48]:


titles = []
images = []
txts = []


# In[51]:


input_rootdir = 'data/birds/CUB_200_2011/graysclae_image'
input_text = 'data/birds/text'
cnt = 0
for subfolder in os.listdir(input_rootdir):
    if subfolder == '.DS_Store':
        continue
    img_ls = os.listdir(input_rootdir+'/' + subfolder)
    if img_ls == '.DS_Store':
        continue
    for img_name in img_ls:
        if img_name == '.DS_Store':
            continue
        img = Image.open(input_rootdir+ '/' + subfolder +'/' + img_name)
        img_tensor = preprocess(img).squeeze(0)
        with open (input_text + '/' + subfolder + '/' + img_name[:-4] + '.txt') as f:
            txt = f.read().replace('\n', '')
            txt = colors_re.sub("", txt)
        titles.append(img_name[:-4])
        images.append(img_tensor)
        txts.append(txt)
        cnt += 1
    print(input_rootdir+ '/' + subfolder , '***', cnt)
    cnt = 0


# In[53]:


with open('title.pkl','wb') as f:
    pickle.dump(titles,f)


# In[54]:


images_tensors = torch.tensor(images)


# In[ ]:


with open('img.pkl','wb') as f:
    pickle.dump(images,f)


# In[ ]:





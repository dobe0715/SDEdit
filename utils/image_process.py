#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
import torchvision.utils as tvu

def preprocess(origin_img : str, stroked_img : str, resolution=256):
    
    transforms = Compose([
        Resize((resolution, resolution)), 
        ToTensor()
    ])
    
    o_img = Image.open(origin_img).convert("RGB")
    s_img = Image.open(stroked_img).convert("RGB")
    
    o_img = transforms(o_img)
    s_img = transforms(s_img)
    
    
    mask = (o_img == s_img).float()
    
    return s_img, mask

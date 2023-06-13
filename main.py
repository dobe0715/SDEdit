


# In[1]:


import easydict
# import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
# import torch.utils.tensorboard as tb
import copy

from runner import Diffusion


# In[2]:


def make_parse_args():
    args = easydict.EasyDict({'seed': 1234, 
                              'exp': 'exp', 
                              'comment': '', 
                              'verbose': 'info', 
                              'sample': 'store_true', 
                              'i': 'images', 
                              'image_folder': 'images', 
                              'ni': 'store_true', 
                              'sample_step': 3, 
                              't': 400})

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args


# In[3]:


args = make_parse_args()
config = "celeba.yml"
origin_img = input("original image url : ")
stroked_img = input("stroked image url : ")

try:
    runner = Diffusion(args, config, origin_img, stroked_img)
    runner.image_editing_sample()
except Exception:
    logging.error(traceback.format_exc())


#!/usr/bin/env python
# coding: utf-8

# In[17]:


import yaml
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torchvision.utils as tvu

from models.diffusion import Model
from utils.diffusion_util import get_beta_schedule, extract, image_editing_denoising_step_flexible_mask
from utils.image_process import preprocess


import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# In[18]:


class Diffusion:
    def __init__(self, args, config, origin_img, stroked_img, device=None):
        self.args = args
        self.config = config
        self.origin_img = origin_img
        self.stroked_img = stroked_img

        with open(self.config, 'r') as f:
            config = yaml.safe_load(f)
            config = dict2namespace(config)

        self.config = config

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas *             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def image_editing_sample(self):
        print("Loading model")

        url = "https://sdediffusion.s3.ap-southeast-2.amazonaws.com/celeba_hq.ckpt"
        ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        config = "celeba.yml"
        
        model = Model(self.config)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0
        
        
        # n = self.config.sampling.batch_size
        n = 4
        model.eval()
        print("Start sampling")
        with torch.no_grad():
            origin_img, stroked_img = self.origin_img, self.stroked_img

            resolution = self.config.data.image_size
            
            img, mask = preprocess(origin_img, stroked_img, resolution=resolution)
            
            mask = mask.to(self.device)
            img = img.to(self.device)

            img = img.unsqueeze(dim=0)
            img = img.repeat(n, 1, 1, 1)
            
            x0 = img            

            tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
            x0 = (x0 - 0.5) * 2.

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))

                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):
                        t = (torch.ones(n) * i).to(self.device)
                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                        logvar=self.logvar,
                                                                        betas=self.betas)
                        x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{it}.png'))
                        progress_bar.update(1)

                x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                torch.save(x, os.path.join(self.args.image_folder,
                                           f'samples_{it}.pth'))
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'samples_{it}.png'))

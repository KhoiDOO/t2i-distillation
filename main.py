import argparse
import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance import Guidance, GuidanceConfig
from tqdm import tqdm
from datetime import datetime

from utils import *

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a DSLR photo of a dolphin")
parser.add_argument("--extra_src_prompt", type=str, default=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed")
parser.add_argument("--extra_tgt_prompt", type=str, default=", detailed high resolution, high quality, sharp")
parser.add_argument("--mode", type=str, default="sds", choices=["bridge", "sds", "nfsd", "vsd", "jsd"])
parser.add_argument("--cfg_scale", type=float, default=40)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_steps", type=int, default=1000)
parser.add_argument("--stage_two_start_step", type=int, default=500)
parser.add_argument("--snap", type=int, default=50)
parser.add_argument("--fps", type=int, default=10)
args = parser.parse_args()

save_dir, debug_dir, cache_dir = setup(args=args)

guidance = Guidance(GuidanceConfig(sd_pretrained_model_or_path="stabilityai/stable-diffusion-2-1-base"), use_lora=(args.mode == "vsd"))

im = torch.randn((1, 4, 64, 64), device=guidance.unet.device)

seed_everything(args.seed)

batch_size = 1

im.requires_grad_(True)
im.retain_grad()

im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
if args.mode == "vsd":
    lora_optimizer = torch.optim.AdamW([{"params": guidance.unet_lora.parameters(), "lr": 3e-4}],weight_decay=0)

im_opts = []

for step in tqdm(range(args.n_steps)):

    guidance.config.guidance_scale = args.cfg_scale
    if args.mode == "bridge":
        if step < args.stage_two_start_step:
            loss_dict = guidance.sds_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
        else:
            loss_dict = guidance.bridge_stage_two(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)

    elif args.mode == "sds":
        loss_dict = guidance.sds_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "nfsd":
        loss_dict = guidance.nfsd_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "vsd":
        loss_dict = guidance.vsd_loss(im=im, prompt=args.prompt, cfg_scale=7.5)
        lora_loss = loss_dict["lora_loss"]
        lora_loss.backward()
        lora_optimizer.step()
        lora_optimizer.zero_grad()
    else:
        raise ValueError(args.mode)

    grad = loss_dict["grad"]
    # src_x0 = loss_dict["src_x0"] if "src_x0" in loss_dict else grad

    im.backward(gradient=grad)
    im_optimizer.step()
    im_optimizer.zero_grad()

    if (step + 1) % args.snap == 0:
        decoded = decode_latent(guidance, im.detach(), device).cpu().numpy()
        im_opts.append(decoded)
        plt.imsave(os.path.join(debug_dir, f"{step + 1}.png"), decoded)

imageio.mimwrite(os.path.join(save_dir, "debug_optimization.mp4"), np.stack(im_opts).astype(np.float32) * 255, fps=args.fps, codec="libx264")
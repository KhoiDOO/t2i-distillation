import argparse
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance import Guidance, GuidanceConfig
from tqdm import tqdm

from utils import *
from stats import *

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="a DSLR photo of a dolphin")
parser.add_argument("--extra_src_prompt", type=str, default=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed")
parser.add_argument("--extra_tgt_prompt", type=str, default=", detailed high resolution, high quality, sharp")
parser.add_argument("--mode", type=str, default="sds", choices=["bridge", "sds", "nfsd", "vsd", "sdsm", "lucid", "lucids", "jsdg", "asd"])
parser.add_argument("--cfg_scale", type=float, default=100)
parser.add_argument("--denoise_cfg_scale", type=float, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=-0.75)
parser.add_argument("--deltat", type=int, default=80)
parser.add_argument("--deltas", type=int, default=200)
parser.add_argument("--numt", type=int, default=2)

parser.add_argument("--didx", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_steps", type=int, default=5000)
parser.add_argument("--stage_two_start_step", type=int, default=500)
parser.add_argument("--snap", type=int, default=100)
parser.add_argument("--fps", type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda", args.didx)

save_dir, debug_dir, cache_dir = setup(args=args)

guidance = Guidance(GuidanceConfig(sd_pretrained_model_or_path="stabilityai/stable-diffusion-2-1-base", device=device), use_lora=(args.mode == "vsd"))

stats_monitor = Stats(run_dir=save_dir)

batch_size = 1

seed_everything(args.seed)

im = torch.randn((batch_size, 4, 64, 64), device=guidance.unet.device)

im.requires_grad_(True)
im.retain_grad()

im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
if args.mode == "vsd":
    lora_optimizer = torch.optim.AdamW([{"params": guidance.unet_lora.parameters(), "lr": 3e-4}],weight_decay=0)

im_opts = []

for step in tqdm(range(args.n_steps)):
    start_time = time.time()
    guidance.config.guidance_scale = args.cfg_scale
    if args.mode == "bridge":
        if step < args.stage_two_start_step:
            loss_dict = guidance.sds_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
        else:
            loss_dict = guidance.bridge_stage_two(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "sds":
        loss_dict = guidance.sds_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "sdsm":
        loss_dict = guidance.sdsm_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "nfsd":
        loss_dict = guidance.nfsd_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
    elif args.mode == "vsd":
        loss_dict = guidance.vsd_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale)
        lora_loss = loss_dict["lora_loss"]
        lora_loss.backward()
        lora_optimizer.step()
        lora_optimizer.zero_grad()
    elif args.mode == "jsdg":
        loss_dict = guidance.jsdg_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, nt=args.numt)
    elif args.mode == "lucids":
        loss_dict = guidance.lucids_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, denoise_cfg_scale=args.denoise_cfg_scale, delta_t=args.deltat)
    elif args.mode == "lucid":
        loss_dict = guidance.lucid_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, denoise_cfg_scale=args.denoise_cfg_scale, delta_t=args.deltat, delta_s=args.deltas)
    elif args.mode =='asd':
        loss_dict = guidance.asd_loss(im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, gamma=args.gamma)
        lora_loss = loss_dict["lora_loss"]
        lora_loss.backward()
        lora_optimizer.step()
        lora_optimizer.zero_grad()
    else:
        raise ValueError(args.mode)
    
    stats_monitor('grad_time', time.time() - start_time)

    grad = loss_dict["grad"]

    stats_monitor('grad_norm', grad_norm(grad=grad))

    start_time = time.time()
    im.backward(gradient=grad)
    stats_monitor('backward_time', time.time() - start_time)
    
    im_optimizer.step()
    im_optimizer.zero_grad()

    if (step + 1) % args.snap == 0:
        decoded = decode_latent(guidance, im.detach(), device).cpu().numpy()
        im_opts.append(decoded)
        plt.imsave(os.path.join(debug_dir, f"{step + 1}.png"), decoded)

stats_monitor.save()

imageio.mimwrite(os.path.join(save_dir, "debug_optimization.mp4"), np.stack(im_opts).astype(np.float32) * 255, fps=args.fps, codec="libx264")
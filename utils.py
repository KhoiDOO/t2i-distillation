import torch
import gc
import os
import json
import random
import numpy as np

from datetime import datetime
from jaxtyping import Float

def grad_norm(grad: Float[torch.Tensor, "..."]):
    return grad.norm(p=2).detach().cpu().item()

def decode_latent(guidance, latent, device):
    latent = latent.detach().to(device)
    with torch.no_grad():
        rgb = guidance.decode_latent(latent)
    rgb = rgb.float().cpu().permute(0, 2, 3, 1)
    rgb = rgb.permute(1, 0, 2, 3)
    rgb = rgb.flatten(start_dim=1, end_dim=2)
    return rgb

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def setup(args):
    save_dir = "results/%s/%s/%s/%s" % (args.mode, args.prompt.replace(" ", "_"), f'seed_{args.seed}', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    debug_dir = os.path.join(save_dir, 'debug')
    cache_dir = '.cache'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print("Save dir:", save_dir)

    config_path = os.path.join(save_dir, 'config.json')

    with open(config_path, 'w') as file:
        json.dump(vars(args), file)

    return save_dir, debug_dir, cache_dir

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class ToWeightsDType(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        return self.module(x).to(self.dtype)

def coeffs(n):
    nums = torch.randn(n)
    return nums / torch.sum(nums)
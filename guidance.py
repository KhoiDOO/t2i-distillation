from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from jaxtyping import Float

from utils import *
from lucid_utils import ddim_step

@dataclass
class GuidanceConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v2-1-base"
    sd_pretrained_model_or_path_lora: str = "stabilityai/stable-diffusion-2-1"

    num_inference_steps: int = 1000
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = ""
    tgt_prompt: str = ""

    guidance_scale: float = 30
    guidance_scale_lora: float = 1.0
    sdedit_guidance_scale: float = 15
    device: torch.device = torch.device("cuda")
    lora_n_timestamp_samples: int = 1

    sync_noise_and_t: bool = True
    lora_cfg_training: bool = True

    device: torch.device = torch.device("cuda")


class Guidance(object):
    def __init__(self, config: GuidanceConfig, use_lora: bool = False):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")

        if use_lora:
            self.pipe_lora = DiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path_lora).to(self.device)
            self.single_model = False
            del self.pipe_lora.vae
            del self.pipe_lora.text_encoder
            cleanup()
            self.vae_lora = self.pipe_lora.vae = self.pipe.vae
            self.unet_lora = self.pipe_lora.unet
            for p in self.unet_lora.parameters():
                p.requires_grad_(False)
            # FIXME: hard-coded dims
            self.camera_embedding = TimestepEmbedding(16, 1280).to(self.device)
            self.unet_lora.class_embedding = self.camera_embedding
            self.scheduler_lora = DDIMScheduler.from_config(self.pipe_lora.scheduler.config)
            self.scheduler_lora.set_timesteps(config.num_inference_steps)
            self.pipe_lora.scheduler = self.scheduler_lora

            # set up LoRA layers
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (None if name.endswith("attn1.processor") else self.unet_lora.config.cross_attention_dim)
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

            self.unet_lora.set_attn_processor(lora_attn_procs)

            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(self.device)
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def encode_text(self, prompt):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt")
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)
    
    def get_variance(self, timestep, scheduler=None):

        if scheduler is None:
            scheduler = self.scheduler

        prev_timestep = (timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    @contextmanager
    def disable_unet_class_embedding(self, unet):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def sample_timestep(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)

        self.min_step = min_step
        self.max_step = max_step
        
        max_step = max(max_step, min_step + 1)
        idx = torch.randint(min_step, max_step, [batch_size], dtype=torch.long, device="cpu")
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev

    def sample_lucids_timestep(self, batch_size, delta_t):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        max_step = max(max_step, min_step + 1)
        
        idx_t = torch.randint(min_step, max_step, [batch_size], dtype=torch.long, device="cpu")
        idx_s = max(idx_t - delta_t, torch.ones_like(idx_t) * 0)
        
        t = timesteps[idx_t].cpu()
        t_prev = timesteps[idx_s].cpu()

        return t, t_prev

    def sample_lucid_timestep(self, batch_size, delta_t, delta_s):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = 1 if self.config.min_step_ratio <= 0 else int(len(timesteps) * self.config.min_step_ratio)
        max_step = len(timesteps) if self.config.max_step_ratio >= 1 else int(len(timesteps) * self.config.max_step_ratio)
        max_step = max(max_step, min_step + 1)
        
        idx_t = torch.randint(min_step, max_step, [batch_size], dtype=torch.long, device="cpu")
        idx_s = max(idx_t - delta_t, torch.ones_like(idx_t) * 0)
        
        t = timesteps[idx_t].cpu()
        t_prev = timesteps[idx_s].cpu()

        n = int(np.ceil(idx_s / delta_s))

        starting_ind = max(idx_s - delta_s * n, torch.ones_like(idx_t) * 0)

        return idx_t, idx_s, t, t_prev, n, starting_ind

    def sds_loss(self, im, prompt=None, cfg_scale=100, noise=None):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)

        if noise is None:
            noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}

    def jsdg_loss(self, im, prompt=None, cfg_scale=100, nt=5):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)

        ts = [self.sample_timestep(batch_size)[0] for i in range(nt)]

        noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        cs = coeffs(nt).to(self.device)

        scores = 0
        for i, _t in enumerate(ts):
            latents_noisy = scheduler.add_noise(im, noise, _t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.unet.forward(latent_model_input, torch.cat([_t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            scores += cs[i] * noise_pred - noise
        
        w = 1 - scheduler.alphas_cumprod[t].to(device)
        alphas = scheduler.alphas_cumprod[t].to(device)
        wp = (((1 - alphas) / alphas) ** 0.5)

        grad = w * (noise_pred - noise) + wp * scores
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}

    def sdsm_loss(self, im, prompt=None, cfg_scale=100, noise=None):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)

        if noise is None:
            noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
        noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * (noise_pred)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}

    def bridge_stage_two(self, im, prompt=None, cfg_scale=30,
        extra_tgt_prompts=", detailed high resolution, high quality, sharp",
        extra_src_prompts=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed",
        noise=None,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt + extra_tgt_prompts, src_prompt=prompt + extra_src_prompts)
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)

        if noise is None:
            noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        text_embeddings = torch.cat([tgt_text_embedding, src_text_embedding], dim=0)
        noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
        noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * cfg_scale * (noise_pred_tgt - noise_pred_src)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}

    def nfsd_loss(self, im, prompt=None, cfg_scale=100):
        device = self.device
        scheduler = self.scheduler

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)

        noise = torch.randn_like(im)

        latents_noisy = scheduler.add_noise(im, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature
        with torch.no_grad():
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            delta_C = cfg_scale * (noise_pred_text - noise_pred_uncond)

        self.update_text_features(tgt_prompt="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy")
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature
        with torch.no_grad():
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            noise_pred = self.unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings).sample
            noise_pred_text_neg, _ = noise_pred.chunk(2)

        delta_D = noise_pred_uncond if t < 200 else (noise_pred_uncond - noise_pred_text_neg)

        w = 1 - scheduler.alphas_cumprod[t].to(device)
        grad = w * (delta_C + delta_D)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}
    
    def lucids_loss(self, im, prompt=None, cfg_scale=100, denoise_cfg_scale=1, delta_t=1):
        device = self.device
        scheduler = self.scheduler

        self.update_text_features(
            src_prompt=prompt,
            tgt_prompt="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation."
        )
        prompt_embedding = self.src_text_feature
        negative_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]

        t, t_prev = self.sample_lucids_timestep(batch_size, delta_t)

        noise = torch.randn_like(im)

        curr_latent = scheduler.add_noise(im, noise, t)
        prev_latent = scheduler.add_noise(im, noise, t_prev)

        curr_latent_input = torch.cat([curr_latent] * 2, dim=0)
        prev_latent_input = torch.cat([prev_latent] * 2, dim=0)

        with torch.no_grad():
            curr_text_embeddings = torch.cat([prompt_embedding, uncond_embedding], dim=0)
            curr_noise_pred = self.unet.forward(curr_latent_input, torch.cat([t] * 2).to(device), encoder_hidden_states=curr_text_embeddings).sample
            noise_pred_prompt, noise_pred_uncond = curr_noise_pred.chunk(2)
            curr_noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_prompt - noise_pred_uncond)
        
            prev_text_embeddings = torch.cat([negative_embedding, uncond_embedding], dim=0)
            prev_noise_pred = self.unet.forward(prev_latent_input, torch.cat([t_prev] * 2).to(device), encoder_hidden_states=prev_text_embeddings).sample
            noise_pred_negative, noise_pred_uncond = prev_noise_pred.chunk(2)
            prev_noise_pred = noise_pred_uncond + denoise_cfg_scale * (noise_pred_negative - noise_pred_uncond)

        alphas = scheduler.alphas_cumprod[t].to(device)
        w = (((1 - alphas) / alphas) ** 0.5)

        grad = w * (curr_noise_pred - prev_noise_pred)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}

    def lucid_loss(self, im, prompt=None, cfg_scale=100, denoise_cfg_scale=1, delta_t=1, delta_s=1):

        device = self.device
        scheduler = self.scheduler

        self.update_text_features(
            src_prompt=prompt,
            tgt_prompt="unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation."
        )
        prompt_embedding = self.src_text_feature
        negative_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        idx_t, idx_s, t, t_prev, n, starting_ind = self.sample_lucid_timestep(batch_size, delta_t, delta_s)
        
        with torch.no_grad():
            
            noise = torch.randn_like(im)

            prev_noisy_lat = scheduler.add_noise(im, noise, t)

            cur_ind_t = starting_ind
            cur_noisy_lat = prev_noisy_lat

            pred_scores = []

            for i in range(n):
                cur_noisy_lat_ = scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.device)

                latent_model_input = torch.cat([cur_noisy_lat_] * 2, dim=0)
                unet_output = self.unet.forward(
                    latent_model_input, 
                    torch.cat([self.timesteps[cur_ind_t]] * 2).to(device), 
                    encoder_hidden_states=torch.cat(
                        [negative_embedding, uncond_embedding], 
                        dim=0
                    )
                ).sample

                negative_cond, null_cond = unet_output.chunk(2)

                unet_output = null_cond + denoise_cfg_scale * (negative_cond - null_cond) #inverse cfg

                pred_scores.append((cur_ind_t, unet_output))

                next_ind_t = min(cur_ind_t + delta_s, idx_s)
                cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
                delta_t_ = next_t-cur_t if isinstance(scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

                cur_noisy_lat = ddim_step(scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, 0.0).prev_sample
                cur_ind_t = next_ind_t

                del unet_output
                torch.cuda.empty_cache()

                if cur_ind_t == idx_s:
                    break
            
            pred_scores_xs = pred_scores[::-1]

            cur_ind_t = idx_s

            cur_noisy_lat_ = scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.device)

            latent_model_input = torch.cat([cur_noisy_lat_] * 2, dim=0)
            unet_output = self.unet.forward(
                latent_model_input, 
                torch.cat([self.timesteps[idx_s]] * 2).to(device), 
                encoder_hidden_states=torch.cat(
                    [negative_embedding, uncond_embedding], 
                    dim=0
                )
            ).sample

            negative_cond, null_cond = unet_output.chunk(2)
            unet_output = null_cond + denoise_cfg_scale * (negative_cond - null_cond) #inverse cfg
            pred_scores = [(cur_ind_t, unet_output)]

            next_ind_t = min(cur_ind_t + delta_t, idx_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = ddim_step(scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, 0.0).prev_sample
            cur_ind_t = next_ind_t

            pred_scores = pred_scores + pred_scores_xs
            target = pred_scores[0][1]

            cur_noisy_lat = scheduler.scale_model_input(cur_noisy_lat, t)
            latent_model_input = torch.cat([cur_noisy_lat_] * 2, dim=0)

            unet_output = self.unet(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=torch.cat(
                    [prompt_embedding, negative_embedding], 
                    dim=0
                )
            ).sample

            promtp_cond, negative_cond = unet_output.chunk(2)
            delta_DSD = promtp_cond - negative_cond
        
        pred_noise = negative_cond + cfg_scale * delta_DSD

        alphas = scheduler.alphas_cumprod[idx_t].to(device)
        w = (((1 - alphas) / alphas) ** 0.5)

        grad = w * (pred_noise - target)
        grad = torch.nan_to_num(grad)
        return {"grad": grad, "t": t}
    
    def vsd_loss(self, im, prompt=None, cfg_scale=7.5):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        camera_condition = torch.zeros([batch_size, 4, 4], device=device)

        with torch.no_grad():
            # random timestamp
            t = torch.randint(20, 980 + 1, [batch_size], dtype=torch.long, device=self.device)

            noise = torch.randn_like(im)

            latents_noisy = scheduler.add_noise(im, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs)

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.unet_lora.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=torch.cat([tgt_text_embedding] * 2),
                class_labels=torch.cat([camera_condition.view(batch_size, -1), camera_condition.view(batch_size, -1)], dim=0),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        (noise_pred_pretrain_text, noise_pred_pretrain_uncond) = noise_pred_pretrain.sample.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + cfg_scale * (noise_pred_pretrain_text - noise_pred_pretrain_uncond)
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(device=latents_noisy.device, dtype=latents_noisy.dtype)
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (noise_pred_est_camera, noise_pred_est_uncond) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.config.guidance_scale_lora * (noise_pred_est_camera - noise_pred_est_uncond)

        w = (1 - scheduler.alphas_cumprod[t.cpu()]).view(-1, 1, 1, 1).to(device)
        grad = w * (noise_pred_pretrain - noise_pred_est)

        grad = torch.nan_to_num(grad)
        loss_lora = self.train_lora_vsd(im, text_embeddings, camera_condition)
        return {"lora_loss": loss_lora, "grad": grad, "t": t}

    def train_lora_vsd(self, latents: Float[torch.Tensor, "B 4 64 64"], 
                       text_embeddings: Float[torch.Tensor, "BB 77 768"], 
                       camera_condition: Float[torch.Tensor, "B 4 4"]):
        scheduler = self.scheduler_lora

        B = latents.shape[0]
        latents = latents.detach().repeat(self.config.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(int(scheduler.config.num_train_timesteps * 0.0), int(scheduler.config.num_train_timesteps * 1.0),
            [B * self.config.lora_n_timestamp_samples], dtype=torch.long, device=self.device)

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler_lora.config.prediction_type}")
        # use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.config.lora_cfg_training and np.random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.unet_lora.forward(noisy_latents, t,
            encoder_hidden_states=text_embeddings_cond.repeat(self.config.lora_n_timestamp_samples, 1, 1),
            class_labels=camera_condition.view(B, -1).repeat(self.config.lora_n_timestamp_samples, 1),
            cross_attention_kwargs={"scale": 1.0},
        ).sample
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    
    def asd_loss(self, im, prompt=None, cfg_scale=7.5, gamma=-0.75):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        camera_condition = torch.zeros([batch_size, 4, 4], device=device)

        with torch.no_grad():
            # random timestamp
            t = torch.randint(20, 980 + 1, [batch_size], dtype=torch.long, device=self.device)

            noise = torch.randn_like(im)

            latents_noisy = scheduler.add_noise(im, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = unet.forward(latent_model_input, torch.cat([t] * 2).to(device), encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs)

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.unet_lora.forward(
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=torch.cat([tgt_text_embedding] * 2),
                class_labels=torch.cat([camera_condition.view(batch_size, -1), camera_condition.view(batch_size, -1)], dim=0),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        (noise_pred_pretrain_text, noise_pred_pretrain_uncond) = noise_pred_pretrain.sample.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + cfg_scale * (noise_pred_pretrain_text - noise_pred_pretrain_uncond)
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(device=latents_noisy.device, dtype=latents_noisy.dtype)
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (noise_pred_est_camera, noise_pred_est_uncond) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.config.guidance_scale_lora * (noise_pred_est_camera - noise_pred_est_uncond)

        w = (1 - scheduler.alphas_cumprod[t.cpu()]).view(-1, 1, 1, 1).to(device)
        grad = w * (noise_pred_pretrain - noise_pred_est)

        grad = torch.nan_to_num(grad)
        loss_lora = self.train_lora_asd(im, text_embeddings, camera_condition, gamma)
        return {"lora_loss": loss_lora, "grad": grad, "t": t}

    def train_lora_asd(self, latents: Float[torch.Tensor, "B 4 64 64"], 
                       text_embeddings: Float[torch.Tensor, "BB 77 768"], 
                       camera_condition: Float[torch.Tensor, "B 4 4"],
                       gamma=-0.75):
        scheduler = self.scheduler_lora

        B = latents.shape[0]
        latents = latents.detach().repeat(self.config.lora_n_timestamp_samples, 1, 1, 1)

        text_embeddings, _ = text_embeddings.chunk(2)

        t = torch.randint(
            int(scheduler.config.num_train_timesteps * 0.0), 
            int(scheduler.config.num_train_timesteps * 1.0),
            [B * self.config.lora_n_timestamp_samples], 
            dtype=torch.long, 
            device=self.device
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)

        with torch.no_grad():
            latent_model_input = noisy_latents

            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = unet.forward(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings.repeat(self.config.lora_n_timestamp_samples, 1, 1),
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

        if self.config.lora_cfg_training and np.random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
            target_text = noise_pred_pretrain.detach()
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
            target_text = self.scheduler.get_velocity(latents, noise_pred_pretrain, t).detach()
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler_lora.config.prediction_type}")
        
        noise_pred = self.unet_lora.forward(
            noisy_latents.detach(), 
            t,
            encoder_hidden_states=text_embeddings.repeat(self.config.lora_n_timestamp_samples, 1, 1),
            class_labels=camera_condition.view(B, -1).repeat(self.config.lora_n_timestamp_samples, 1),
            cross_attention_kwargs={"scale": 1.0},
        ).sample

        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean") + \
            ((gamma) * F.mse_loss(noise_pred.float(), target_text.float(), reduction="mean"))
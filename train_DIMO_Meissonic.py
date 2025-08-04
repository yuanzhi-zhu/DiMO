import os
import gc
import sys
import logging
import inspect
import argparse
import datetime
import subprocess
from packaging import version

from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict, Tuple

import math
import numpy as np
import scipy.stats as stats
import random
import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers import EMAModel, VQModel

from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, broadcast
import shutil
import copy

from Meissonic_src.dataset_utils import compute_embeddings
from Meissonic_src.scheduler import Scheduler
from Meissonic_src.transformer import Transformer2DModel
from Meissonic_src.pipeline import Pipeline

from typing import Optional
from contextlib import contextmanager

EXT = "png"

# ----------------------------------------------- utils -----------------------------------------------

def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]

def save_code_snapshot(model_path):
    os.makedirs(model_path, exist_ok=True)
    for f in get_file_list():
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(model_path, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(model_path, f))


def filter_nan_mask(images, pred_real=None, pred_fake=None, weight=None):
    nan_mask_imgs = torch.isnan(images).flatten(start_dim=1).any(dim=1)
    nan_mask_real = torch.isnan(pred_real).flatten(start_dim=1).any(dim=1) if pred_real is not None else torch.zeros_like(nan_mask_imgs)
    nan_mask_fake = torch.isnan(pred_fake).flatten(start_dim=1).any(dim=1) if pred_fake is not None else torch.zeros_like(nan_mask_imgs)
    nan_mask = nan_mask_imgs | nan_mask_real | nan_mask_fake
    # Check if there are any NaN values present
    if nan_mask.any():
        # Invert the nan_mask to get a mask of samples without NaNs
        non_nan_mask = ~nan_mask
        # Filter out samples with NaNs from pred_real and pred_fake
        images = images[non_nan_mask]
        pred_real = pred_real[non_nan_mask] if pred_real is not None else None
        pred_fake = pred_fake[non_nan_mask] if pred_fake is not None else None
        weight = weight[non_nan_mask] if weight is not None else None
    return images, pred_real, pred_fake, weight


def save_rng_state(cuda: bool = True):
    """
    Save the states of all.
    """
    rng_state_torch = torch.get_rng_state()
    rng_state_numpy = np.random.get_state()
    rng_state_random = random.getstate()
    rng_state_cuda: Optional[torch.Tensor] = None
    if cuda and torch.cuda.is_available():
        rng_state_cuda = torch.cuda.get_rng_state()
    rng_state_list = [rng_state_torch, rng_state_numpy, rng_state_random, rng_state_cuda]
    return rng_state_list

def restore_rng_state(rng_state_list):
    """
    Restore the states of all.
    """
    rng_state_torch, rng_state_numpy, rng_state_random, rng_state_cuda = rng_state_list
    torch.set_rng_state(rng_state_torch)
    np.random.set_state(rng_state_numpy)
    random.setstate(rng_state_random)
    if torch.cuda.is_available() and rng_state_cuda is not None:
        torch.cuda.set_rng_state(rng_state_cuda)

@contextmanager
def random_seed_context(seed, cuda: bool = True):
    """
    Context manager for handling random number generator states.
    Saves the current state of torch, numpy, and random RNGs,
    and restores them upon exit.

    Args:
        cuda (bool): Whether to also save/restore CUDA RNG state.
                    Default is False.
    """
    try:
        # Save all RNG states
        rng_state_list = save_rng_state(cuda)
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        yield
        
    finally:
        # Restore all RNG states
        restore_rng_state(rng_state_list)


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99, global_step=None):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    if global_step is not None:
        # rate = 0.9999910000899991 when global_step = 1e6
        # rate = 0.9999100089991001 when global_step = 1e5
        # rate = 0.999100899100899 when global_step = 1e4
        rate = min(rate, (1 + global_step) / (10 + global_step))
        # ema_rampup_ratio = 0.05
        # inv_rate = np.log(rate) / np.log(0.5)
        # rate = 0.5 ** max(inv_rate, 1/(global_step*ema_rampup_ratio))
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.to(targ.device), alpha=1 - rate)

def _prepare_latent_image_ids(batch_size, height, width, device):
    # Prepare latent image ids for RoPE
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    # latent_image_ids = latent_image_ids.unsqueeze(0).repeat(batch_size, 1, 1)
    return latent_image_ids.to(device=device)


# ----------------------------------------------- utils dataloader -----------------------------------------------

def get_batch_generator(dataloader):
    while True:
        for batch in dataloader:
            # DEBUG: never drop text
            # assert cfg_random_null_text_ratio == 0
            # if cfg_random_null_text:
            #     batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
            yield batch


# ----------------------------------------------- utils pred_x0 -----------------------------------------------

# used for true and fake model
def cfg_pred_model(model, code, encoder_hidden_states, cond_embeds, \
                    img_ids, txt_ids, micro_conds, mask_prob=0.5, \
                    uncond_encoder_hidden_states=None, uncond_cond_embeds=None, \
                    codebook_size=8192, cfg=1.):
    bsz = code.shape[0]
    mask_prob = torch.tensor([mask_prob], dtype=torch.long) if type(mask_prob) is float else mask_prob.long()
    cfg = torch.tensor([cfg]).repeat(bsz) if type(cfg) in [int, float] else cfg
    cfg = cfg.view(-1,1,1).to(code.device)
    logits = model(hidden_states=code, # should be (batch size, height, width)
                    encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                    micro_conds=micro_conds, # 
                    pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    timestep= mask_prob.to(device=code.device) * 1000,
                    ).reshape(bsz, codebook_size, -1).permute(0, 2, 1)       # [B, P^2, V]
    if (cfg != 1).any():
        logit_u = model(hidden_states=code, # should be (batch size, height, width)
                        encoder_hidden_states=uncond_encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                        micro_conds=micro_conds, # 
                        pooled_projections=uncond_cond_embeds, # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        timestep=mask_prob.to(device=code.device) * 1000,
                        ).reshape(bsz, codebook_size, -1).permute(0, 2, 1)       # [B, P^2, V]
        # Classifier Free Guidance
        logits = logit_u + cfg * (logits - logit_u)
    return logits

def get_mask_code(code, mode="arccos", value=None, codebook_size=8192, r_min=0.02, r_max=0.98):
    """ Replace the code token by *value* according the the *mode* scheduler
        :param
        code  -> torch.LongTensor(): bsize * patch_size * patch_size, the unmasked code
        mode  -> str:                the rate of value to mask
        value -> int:                mask the code by the value
        :return
        masked_code -> torch.LongTensor(): bsize * patch_size * patch_size, the masked version of the code
        mask        -> torch.LongTensor(): bsize * patch_size * patch_size, the binary mask of the mask
    """
    bsz = code.size(0)
    r = torch.rand(bsz)
    patch_size = code.size(1)
    # get the mask ratio
    if mode == "linear":                # linear scheduler
        val_to_mask = r
    elif mode == "square":              # square scheduler
        val_to_mask = (r ** 2)
    elif mode == "cosine":              # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":              # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    elif mode == "mage":                # mage scheduler
        # # mask ratio mean=0.75, std=0.5, clamp=[0.5,1]
        # val_to_mask = torch.clamp(torch.normal(0.75, 0.25, size=(bsz,)), min=0.5, max=1.)
        # # val_to_mask = torch.clamp(torch.normal(0.7, 0.25, size=(bsz,)), min=0.42, max=0.98)
        mask_ratio_min = 0.5
        mask_ratio_max = 1.
        mask_ratio_mu = 0.55
        mask_ratio_std = 0.25
        mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
        val_to_mask = torch.tensor(mask_ratio_generator.rvs(bsz))
    else:
        val_to_mask = None
    val_to_mask = val_to_mask.clamp(min=r_min, max=r_max)

    ### Forward discrete diffusion:
    # # stochastic version
    # mask = torch.rand((bsz, seq_len), device=code.device) < val_to_mask
    # mask_code = torch.where(mask, value, code)
    # exact version
    mask_code = code.detach().clone()
    # Sample the amount of tokens + localization to mask
    num_token_masked = (patch_size**2 * val_to_mask).round().clamp(min=1).long()  # Convert to integer
    batch_randperm = torch.rand(bsz, patch_size**2).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask.reshape(bsz, patch_size, patch_size)

    assert torch.equal(mask.sum(dim=(1,2)), num_token_masked)

    if value > 0:  # Mask the selected token by the value
        mask_code[mask] = torch.full_like(mask_code[mask], value)
    else:  # Replace by a randon token
        mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

    return mask_code, mask, val_to_mask


# ----------------------------------------------- log_validation -----------------------------------------------

@torch.no_grad()
def log_validation(val_pipeline, model, vae, accelerator, global_step, \
                    fixed_ratio=0.5, resolution=1024, patch_size=64, codebook_size=8192, mask_value=8255, \
                    gen_temp=1., cfg=9, infer_step=64, local_seed=42, name="model", log_dir='', \
                    micro_conds=None, img_ids=None, txt_ids=None, \
                    top_k=0, top_p=0., noise_emb_perturb=0., num_images_per_prompt=2,
                    ):
    with random_seed_context(local_seed, cuda=True):
        torch.cuda.empty_cache()
        micro_conds = micro_conds[0].unsqueeze(0).repeat(num_images_per_prompt, 1) if micro_conds is not None else None
        vae_dtype = vae.dtype

        validation_prompts = [
            "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
            "a red ball on top of a blue cube, on the ground",
            "front view of a boat sailing in a cup of water",
            "could not found the image",
            "There is no image here to provide a description for.",
            "nine pieces of art in a 3x3 grid, including objects, portraits, scenes, landscapes, and abstract art",
            "欲买桂花同载酒，终不似，少年游。",
        ]
        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        # negative_prompt = ""

        transform = transforms.Compose([
        transforms.Lambda(lambda img: torchvision.transforms.functional.resize(img, (512, 512), antialias=True)),
        transforms.ToTensor()
        ])

        samples = []
        for idx, prompt in enumerate(validation_prompts):
            generator = torch.Generator(device=accelerator.device).manual_seed(local_seed) if local_seed is not None else None
            # Generate sample for visualization
            if infer_step == 1:
                cfg = 1
                # Get the text embedding
                empty_embeds, empty_clip_embeds = compute_embeddings(val_pipeline.text_encoder, val_pipeline.tokenizer, prompt, device=accelerator.device)
                encoder_hidden_states = empty_embeds.repeat(num_images_per_prompt, 1, 1)
                cond_embeds = empty_clip_embeds.repeat(num_images_per_prompt, 1)
                images = sample_one_step(accelerator.unwrap_model(model), vae, encoder_hidden_states=encoder_hidden_states, cond_embeds=cond_embeds, fixed_ratio=fixed_ratio,
                                                patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=accelerator.device,
                                                micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                                top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=noise_emb_perturb)
                resize_transfrom = transform.transforms[0]
                images = [resize_transfrom(img).unsqueeze(0) for img in images]
                samples += images
            else:
                images = val_pipeline(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, \
                                        guidance_scale=cfg, num_inference_steps=infer_step, \
                                        generator=generator, num_images_per_prompt=num_images_per_prompt).images
                images = [transform(img).unsqueeze(0) for img in images]
                samples += images
        samples = make_grid(torch.concat(samples, dim=0), nrow=4)
        Path(f"{log_dir}/step_{global_step}").mkdir(parents=True, exist_ok=True)
        save_path = f"{log_dir}/step_{global_step}/{name}-infer_step{infer_step}-cfg{cfg}-seed_{local_seed}.{EXT}"
        save_image(samples, save_path, normalize=False)

        gc.collect()
        torch.cuda.empty_cache()
        vae.to(vae_dtype) #!!!: no idea why the dtype of the vae outside this function also changed. So, have to change it back.
        # logging.info(f"finsihed validation.... samples saved to {log_dir}/step_{global_step}\n")


def generate_one_step(model, encoder_hidden_states=None, cond_embeds=None, fixed_ratio=0.5, 
                        patch_size=64, codebook_size=8192, mask_value=8255, device='cuda',
                        micro_conds=None, img_ids=None, txt_ids=None,
                        top_k=0, top_p=0., gen_temp=1., noise_emb_perturb=0.):
    bsz = encoder_hidden_states.shape[0]
    # 1. initial random code
    shape_code = torch.ones(bsz, patch_size, patch_size)
    code = torch.randint_like(shape_code, 0, codebook_size).to(device, dtype=torch.long)
    # 2. add mask to fixed ratio
    masked_code = code.detach().clone()
    # Sample the amount of tokens + localization to mask
    # mask = torch.rand(size=code.size()) < fixed_ratio
    # DEBUG: use fixed number of masked tokens
    num_token_masked = torch.tensor(patch_size**2 * fixed_ratio).round().clamp(min=1)
    batch_randperm = torch.rand(bsz, patch_size**2).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask.reshape(bsz, patch_size, patch_size)
    masked_code[mask] = torch.full_like(masked_code[mask], mask_value)
    # 3. one-step generator prediction
    pred_logit = model(hidden_states=masked_code, # should be (batch size, height, width)
                        encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                        micro_conds=micro_conds, # 
                        pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        timestep=torch.tensor([fixed_ratio], device=device, dtype=torch.long) * 1000,
                        noise_emb_perturb=noise_emb_perturb,
                        ).reshape(code.shape[0], codebook_size, -1).permute(0, 2, 1)       # [B, P^2, V]
    
    # Apply temperature scaling
    logits = pred_logit / gen_temp

    # Apply top-k filtering
    if top_k > 0:
        # Set all but the top-k tokens to -infinity
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    # Apply top-p (nucleus) filtering
    if top_p > 0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Scatter sorted indices back to original positions
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)      # [B, P^2, V]
    pred_distri = torch.distributions.Categorical(probs=probs)
    pred_code = pred_distri.sample()
    pred_code = pred_code.view(bsz, patch_size, patch_size)     # [B, P, P]

    return pred_logit, pred_code


@torch.no_grad()
def sample_one_step(model, vae, encoder_hidden_states=None, cond_embeds=None, fixed_ratio=0.5,
                        patch_size=64, codebook_size=8192, mask_value=8255, device='cuda',
                        micro_conds=None, img_ids=None, txt_ids=None, 
                        top_k=0, top_p=0., gen_temp=1., noise_emb_perturb=0.):
    model.eval()
    _, code = generate_one_step(model, encoder_hidden_states, cond_embeds, fixed_ratio=fixed_ratio,
                        patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                        micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                        top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=noise_emb_perturb)
    # decode the final prediction
    _code = torch.clamp(code, 0,  codebook_size-1)
    images = []
    for idx in range(_code.shape[0]):
        x = vae.decode(
            _code[idx:idx+1],
            force_not_quantize=True,
            shape=(
                1,
                patch_size,
                patch_size,
                vae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
        images.append(x)
    images = torch.cat(images)
    model.train()
    return images



def get_pert(noise_emb_perturb, current_training_step) -> float:
    """Randomness schedule for randomness at the current point in training."""
    # if noise_emb_perturb is a number, use it as fixed randomness
    if 'fix' in noise_emb_perturb:
        emb_randomness = float(noise_emb_perturb.split('_')[1])
    elif 'warmup' in noise_emb_perturb:     # E.g., warmup_exp_10000
        warmup_iters = int(noise_emb_perturb.split('_')[-1])
        if 'exp' in noise_emb_perturb:
            # use exponential warmup, from 0 to 1
            emb_randomness = 1 - np.exp(-current_training_step / warmup_iters)
        elif 'cos' in noise_emb_perturb:    # E.g., warmup_cos_10000
            # use cosine warmup, from 0 to 1
            emb_randomness = 1 - (1 + np.cos(np.pi * min(1, current_training_step / warmup_iters))) / 2
            # emb_randomness = 1 - np.cos(np.pi / 2 * min(1, current_training_step / self.warmup_iters))
        else:                        # E.g., warmup_linear_10000
            # use linear warmup, from 0 to 1
            emb_randomness = min(1, current_training_step / warmup_iters)
    else:
        raise NotImplementedError(f'x0_randomness {noise_emb_perturb} Not implemented')
    return emb_randomness

# ----------------------------------------------- main -----------------------------------------------

def main(
    name: str,
    report_to: str,
    
    output_dir: str,
    pretrained_model_name_or_path: str,
    text_encoder_architecture: str = 'open_clip',
    pretrained_model_architecture: str = 'Meissonic',

    wandb_project: str = "",
    wandb_user: str = "",
    data_path: str = "",
    resolution: int = 1024,

    ema_decay: float = 0.9999,
    ema_cpu: bool = False,
    ema_freq: int = 1,
    
    max_train_steps: int = 100000,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    metric_type: str = "fid",   # ["fid", "clip", "fid_clip", ""]
    metric_steps: int = 0,
    metric_steps_tuple: Tuple = (-1,),
    metric_prompts: str = "",

    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler_type: str = "constant",

    num_workers: int = 16,
    train_batch_size: int = 64,
    loss_reduction: str = "mean",
    optimizer_type: str = "adamw",
    adam_beta1: float = 0.9,  # pytorch's default
    adam_beta2: float = 0.999,  # pytorch's default
    adam_weight_decay: float = 1e-2,  # pytorch's default
    adam_epsilon: float = 1e-08,  # pytorch's default
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_steps: int = -1,

    mixed_precision: str = "bf16",
    enable_xformers_memory_efficient_attention: bool = False,

    global_seed: int = 42,
    is_debug: bool = False,

    cast_models_to_dtype: bool = False,
    checkpoints_total_limit: int = 5,
    checkpointing_steps_tuple: list = [5000],
    resume_from_checkpoint_path: str = "",

    no_progress_bar: bool = False,

    # MIM
    # codebook_size: int = 1024,

    # generator
    generator_lr: float = 1e-5,
    fake_rounds: int = 2,

    # DMD
    dm_loss_weight: float = 1,
    distil_loss_type: str = 'RKL',
    fixed_ratio: float = 0.25,
    top_k: int = 0,
    top_p: float = 0.,
    true_cfg: float = 3,
    fake_cfg_eval: float = 1.,
    fake_cfg_train: float = 1.,
    fake_lr: float = 1e-5,
    fake_cfg_drop_ratio = 0.,
    gen_temp: float = 1.,
    true_temp: float = 1.,
    fake_temp: float = 1.,
    ignore_index: int = -100,
    ratio_mode: str = 'arccos',
    ratio_mode_fake: str = 'arccos',
    temperature_fake: float = 1.,
    alpha_fake: float = 1.,
    noise_emb_perturb: str = 'fix_0',
    fix_emb_layer: bool = False,
    adaptive_cfg: bool = False,
    distil_neg_prompt: bool = False,

    # sampling
    sched_mode: str = 'arccos',
    sampling_step: int = 64,
    # mask_value: int = 8255,
    cfg_w: float = 3,
    r_temp: float = 4.5,
    sm_temp: float = 1.,
    ratings: str = 'all',
):
    check_min_version("0.30.3")

    fake_cfg_drop_ratio = fake_cfg_drop_ratio if (fake_cfg_train !=1 or fake_cfg_eval != 1) else 0.
    adam_epsilon = 1e-6 if '16' in mixed_precision else adam_epsilon
    if 'Jeffreys' in distil_loss_type:    # mixtures of FKL and RKL
        Jeffreys_beta = float(distil_loss_type.split('_')[-1])
        if Jeffreys_beta == 0:
            distil_loss_type = 'FKL'
        elif Jeffreys_beta == 1:
            distil_loss_type = 'RKL'

    # Logging folder
    run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")
    name = name + f'DDMD'
    name = name + f'-GPUS{torch.cuda.device_count()}-bs{train_batch_size}'
    name = name + f'-grad_accu{gradient_accumulation_steps}' if gradient_accumulation_steps > 1 else name
    name = name + f'-glr{generator_lr}-flr{fake_lr}'
    name = name + f'-{distil_loss_type}-fix_r{fixed_ratio}-emb_pert_{noise_emb_perturb}'
    name = name + f'-tcfg{true_cfg}-fcfg{fake_cfg_eval}-fcfgt{fake_cfg_train}' if (dm_loss_weight > 0) else name
    name = name + f'-r_mode_{ratio_mode}-r_mode_f_{ratio_mode_fake}'
    # name = name + f'-gen_temp{gen_temp}-true_temp{true_temp}-fake_temp{fake_temp}'
    # name = name + f'-topk{top_k}-topp{top_p}'
    name = name + f'-f_round{fake_rounds}' if fake_rounds > 1 else name
    name = name + f'-a_fake{alpha_fake}' if alpha_fake != 0 else name
    name = name + f'-reduce_{loss_reduction}'
    name = name + f'-seed{global_seed}'
    # name = name + f'-ema{ema_decay}' if ema_decay > 0 else name
    precision = 'fp32' if mixed_precision == 'no' else mixed_precision
    name = name + f'-{precision}'
    name = name + f'-resumed' if resume_from_checkpoint_path else name
    name = name + f'-aesthetic-{ratings}' if ratings else name
    folder_name = "debug" if is_debug else f'{name}/{run_id}'
    output_name = output_dir
    output_dir = os.path.join(f'outputs/{output_dir}', folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    
    report_to = 'wandb' if wandb_project else report_to
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=os.path.join('./outputs', f'{report_to}/{output_name}'))

    accelerator = Accelerator(
        # gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        # split_batches=True,  # It's important to set this to True when using webdataset to get the right number of steps for lr scheduling. If set to False, the number of steps will be devide by the number of processes assuming batches are multiplied by the number of processes
    )
    
    *_, config = inspect.getargvalues(inspect.currentframe())
    config = {k: v for k, v in config.items() if not (k == '_' or k == 'config' or k.startswith('accelerator'))}
    
    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        save_code_snapshot(Path(output_dir, 'codes'))
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples_val", exist_ok=True)
        os.makedirs(f"{output_dir}/meta_checkpoints", exist_ok=True)
        # os.makedirs(f"{output_dir}/samples_metric", exist_ok=True)
        os.makedirs(f"./outputs/{report_to}/{folder_name}", exist_ok=True) if wandb_project else None
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if wandb_project:
            accelerator.init_trackers(
                project_name=wandb_project,
                config=config,
                init_kwargs={"wandb": {"entity": wandb_user,
                                        "name": folder_name,
                                        "dir": f"./outputs/{report_to}/{folder_name}",
                                        },
                            },
                )
        else:   # tensorboard by default
            # tracker_config = dict(vars(args))
            # accelerator.init_trackers(tracker_project_name, config=tracker_config)
            accelerator.init_trackers(f'{folder_name}')

    accelerator.wait_for_everyone()
    logger = get_logger(__name__)
    # Basic configuration for logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ]
    )
    
    logger.info(f'command lines: {str(sys.argv)}')
    logger.info(accelerator.state, main_process_only=False)
    
    # If passed along, set the training seed now.
    if global_seed is not None:
        set_seed(global_seed + accelerator.process_index)
        logger.info(f"Set seed {global_seed + accelerator.process_index} on rank {accelerator.process_index}", main_process_only=False)
    # more efficient when 1. input sizes (batch size, image size, etc.) stay constant 2. running many iterations with the same model architecture
    # Output of the model (MIM and diffusion) will be different for different precision setting here and the general precision (fp32, fp16, bf16)
    torch.backends.cudnn.benchmark = True
    # FIXME Increases numerical precision --> cost more than double the time
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    # Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    if text_encoder_architecture == "open_clip":
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        # text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")    # DEBUG: using original text enc for stable sampling
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    else:
        raise ValueError(f"Unknown text encoder architecture: {text_encoder_architecture}")

    vqvae = VQModel.from_pretrained(pretrained_model_name_or_path, subfolder="vqvae")
    vae_scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    patch_size = resolution // vae_scale_factor     # Load VQGAN patch size

    # Load teacher model
    if pretrained_model_architecture == "Meissonic":
        teacher_model = Transformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer")
    else:
        raise ValueError(f"Unknown model architecture: {pretrained_model_architecture}")
    codebook_size = teacher_model.config.codebook_size
    mask_value = teacher_model.config.vocab_size - 1

    # create a copy of the teacher_model for the generator
    generator = copy.deepcopy(teacher_model)
    # generator = torch.compile(generator)    # TODO: load compile the generator model

    # Freeze vae and models
    text_encoder.to(accelerator.device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vqvae.requires_grad_(False)
    vqvae.encoder = None    # DEBUG: remove the encoder to save memory, as our method is data-free
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(generator).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(generator).dtype}. {low_precision_error_string}"
        )

    generator.train().requires_grad_(True)
    if fix_emb_layer:
        generator.embed.requires_grad_(False)
        generator.mlm_layer.requires_grad_(False)

    if dm_loss_weight > 0:
        fake_model = copy.deepcopy(teacher_model)
        if cast_models_to_dtype:
            fake_model.to(dtype=weight_dtype)
        fake_model.to(accelerator.device)
        if gradient_checkpointing:
            fake_model.gradient_checkpointing = True
            fake_model.enable_gradient_checkpointing()
        fake_model.train().requires_grad_(True)
        if fix_emb_layer:
            fake_model.embed.requires_grad_(False)
            fake_model.mlm_layer.requires_grad_(False)
    else:
        logger.info(f"dm_loss_weight {dm_loss_weight}")

    # Move model and vae to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vqvae.to(accelerator.device, dtype=torch.float32)

    if cast_models_to_dtype:
        # FIXME: cast_models_to_dtype = False --> converge faster but may crash
        # vae.to(dtype=weight_dtype)
        teacher_model.to(dtype=weight_dtype)
        generator.to(dtype=weight_dtype)
    teacher_model.to(accelerator.device)
    generator.to(accelerator.device)

    if ema_decay > 0:
        ema_model = copy.deepcopy(generator)
        ema_model.eval().requires_grad_(False)
        ema_model = accelerator.prepare(ema_model)
        ema_model = ema_model.to('cpu' if ema_cpu else accelerator.device)

    # Enable gradient checkpointing # DEBUG: seems necessary to avoid OOM
    if gradient_checkpointing:
        teacher_model.gradient_checkpointing = True
        teacher_model.enable_gradient_checkpointing()
        generator.gradient_checkpointing = True
        generator.enable_gradient_checkpointing()

    # Enable xformers optimizations
    # FIXME: xformers doesnot help for now
    # possible reason: diffusers has inherent memory efficient attention from pytorch 2.2 https://github.com/huggingface/diffusers/issues/8873
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                ValueError(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            teacher_model.enable_xformers_memory_efficient_attention()
            if dm_loss_weight > 0:
                fake_model.enable_xformers_memory_efficient_attention()
            generator.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # after enabling gradient checkpointing
    teacher_model, generator = accelerator.prepare(teacher_model, generator)

    # Handle saving and loading of checkpoints
    def save_all_state(output_path, save_accelerator_state=False, global_step=0):
        # NOTE: we save the checkpoints to two places 
        # output_path: save the latest one, this is assumed to be a permanent storage
        if save_accelerator_state:
            accelerator.save_state(output_path)
            additional_state = {"global_step": global_step}
            accelerator.save(additional_state, f'{output_path}/additional_state.pth')
        else:   # save the model state_dict
            # accelerator.unwrap_model(fake_model).save_pretrained(os.path.join(output_path, "fake_model"))
            accelerator.unwrap_model(generator).save_pretrained(os.path.join(output_path, "generator"))
            if ema_decay > 0:
                ema_model.save_pretrained(os.path.join(output_path, "ema_model"))

    optimizer_class = torch.optim.AdamW if optimizer_type == "adamw" else torch.optim.Adam
    trainable_params = list(filter(lambda p: p.requires_grad, generator.parameters()))
    logger.info(f"### trainable params number in generator: {sum(p.numel() for p in trainable_params)}")
    optimizer_g = optimizer_class(
        trainable_params,
        lr=generator_lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay if optimizer_type == "adamw" else 0,
        eps=adam_epsilon,
    )
    if dm_loss_weight > 0:
        fake_model = accelerator.prepare(fake_model)
        trainable_params_fake = list(filter(lambda p: p.requires_grad, fake_model.parameters()))
        logger.info(f"### trainable params number in fake_model: {sum(p.numel() for p in trainable_params_fake)}")
        optimizer_fake = optimizer_class(
            trainable_params_fake,
            lr=fake_lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay if optimizer_type == "adamw" else 0,
            eps=adam_epsilon,
        )

    # prepare the optimizer and dataloader
    optimizer_g = accelerator.prepare(optimizer_g)
    if dm_loss_weight > 0:
        optimizer_fake = accelerator.prepare(optimizer_fake)

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint_path:
        logger.info(f"Resuming from checkpoint {resume_from_checkpoint_path}")
        accelerator.load_state(resume_from_checkpoint_path)
        lr_warmup_steps = 0
        additional_state = torch.load(f'{resume_from_checkpoint_path}/additional_state.pth')
        global_step = additional_state["global_step"]
    else:
        global_step = 0

    if scale_lr:
        generator_lr = (generator_lr * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)
        fake_lr = (fake_lr * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    # lr scheduler
    lr_scheduler_g = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer_g,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    if dm_loss_weight > 0:
        lr_scheduler_fake = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer_fake,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps,
        )

    ### dataset creation
    if "LAION_recap_0729" in data_path:
        logger.info(f"Using laion_local dataset")
        from Meissonic_src.laion_local import make_train_dataset
    else:
        logger.info(f"Using aesthetics prompts dataset")
        from Meissonic_src.aesthetics_dataset import make_train_dataset

    train_dataset = make_train_dataset(
        train_data_path=data_path, 
        size = resolution,
        tokenizer=None,
        cfg_drop_ratio = 0,
        rank=accelerator.state.process_index, 
        world_size=accelerator.state.num_processes,
        shuffle=True,
        ratings=ratings,
    )
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        collate_fn=train_dataset.collect_fn if hasattr(train_dataset, 'collect_fn') else None,
    )

    bsz = train_batch_size
    
    # Create uncond embeds for classifier free guidance
    # uncond_prompt_embeds = torch.zeros(train_batch_size, MAX_SEQ_LENGTH, 2048).to(accelerator.device)
    with torch.no_grad():
        empty_embeds, empty_clip_embeds = compute_embeddings(text_encoder, tokenizer, "", text_encoder_architecture, accelerator.device)
        # uncond_encoder_hidden_states, uncond_cond_embeds in shape (bsz, seq_len, embed_dim) and (bsz, proj_dim)
        uncond_encoder_hidden_states, uncond_cond_embeds = empty_embeds.repeat(train_batch_size, 1, 1), empty_clip_embeds.repeat(train_batch_size, 1)
        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        negative_prompt = negative_prompt if distil_neg_prompt else ""
        neg_embeds, neg_clip_embeds = compute_embeddings(text_encoder, tokenizer, negative_prompt, text_encoder_architecture, accelerator.device)
        neg_encoder_hidden_states, neg_cond_embeds = neg_embeds.repeat(train_batch_size, 1, 1), neg_clip_embeds.repeat(train_batch_size, 1)
        micro_conds = torch.tensor([1024, 1024, 0, 0, 6.0])     # [orig_width, orig_height, c_top, c_left, hps_score]
        micro_conds = micro_conds.unsqueeze(0).expand(train_batch_size, -1).to(accelerator.device)
        if resolution == 1024: # only stage 3 and stage 4 do not apply 2*
            img_ids = _prepare_latent_image_ids(bsz, patch_size, patch_size, accelerator.device)
        else:
            img_ids = _prepare_latent_image_ids(bsz, 2*patch_size, 2*patch_size, accelerator.device)
        txt_ids = torch.zeros(empty_embeds.shape[1], 3).to(device=accelerator.device)
    
    # sanity check validation
    if accelerator.is_main_process and validation_steps:
        scheduler = Scheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        if text_encoder_architecture == "CLIP" or text_encoder_architecture == "open_clip":
            val_pipeline = Pipeline(
                transformer=accelerator.unwrap_model(teacher_model),
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vqvae=vqvae,
                scheduler=scheduler,
            )
        val_pipeline = val_pipeline.to(accelerator.device)
        val_pipeline.set_progress_bar_config(disable=True)
        log_validation(val_pipeline, generator, vqvae, accelerator, 0, infer_step=1, fixed_ratio=fixed_ratio, patch_size=patch_size, local_seed=global_seed, codebook_size=codebook_size, mask_value=mask_value, micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids, name="generator", log_dir=f"{output_dir}/samples_val", noise_emb_perturb=0.3)
        log_validation(val_pipeline, teacher_model, vqvae, accelerator, 0, infer_step=sampling_step, local_seed=global_seed, name="teacher_model", log_dir=f"{output_dir}/samples_val")
        # log_validation(val_pipeline, fake_model, vqvae, accelerator, 0, infer_step=sampling_step, local_seed=global_seed, name="fake_model", log_dir=f"{output_dir}/samples_val")
        # log_validation(val_pipeline, generator, vqvae, accelerator, 0, infer_step=sampling_step, local_seed=global_seed, name="generator", log_dir=f"{output_dir}/samples_val")
    
    # Create the dataloader generator once
    batch_gen = get_batch_generator(train_dataloader)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num iters = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  number of processes = {accelerator.num_processes}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    #################################################################################
    #                                 training loop                                 #
    #################################################################################
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=(not accelerator.is_main_process) or no_progress_bar)
    progress_bar.set_description("Steps")

    loss_dict = {}
    loss_dict["loss_generator"] = 0.0
    # DMD loss
    loss_dict["loss_dm"] = 0.0
    loss_dict["loss_fake"] = 0.0
    loss_dict["entropy_bonus"] = 0.0
    loss_dict['score_diff'] = 0.0

    # generator.eval().requires_grad_(False)
    # fake_model.eval().requires_grad_(False)
    for step in range(max_train_steps * fake_rounds + 1):
        ### >>>> Training >>>> ###
        cur_noise_emb_perturb = get_pert(noise_emb_perturb, step)
        #################################################################################
        #                               update generator                                #
        #################################################################################

        COMPUTE_GENERATOR_GRADIENT = step % fake_rounds == 0
        if COMPUTE_GENERATOR_GRADIENT:
            # generator.train().requires_grad_(True)
            optimizer_g.zero_grad(set_to_none=True)
            for accu_iter in range(gradient_accumulation_steps):
                # # get the batch
                batch = next(batch_gen)
                with torch.no_grad():
                    encoder_hidden_states, cond_embeds = compute_embeddings(text_encoder, tokenizer, batch["input_ids"], text_encoder_architecture, accelerator.device)
                pred_logit, pred_code = generate_one_step(generator, encoder_hidden_states, cond_embeds, 
                                                fixed_ratio=fixed_ratio, device=accelerator.device,
                                                micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                                patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value,
                                                top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=cur_noise_emb_perturb)
                # --- DMD loss ---
                if dm_loss_weight > 0:
                    # dmd_mask_code, dmd_mask, dmd_mask_ratio = get_mask_code(pred_code, mode=ratio_mode, value=mask_value, codebook_size=codebook_size, r_max=0.95)
                    dmd_mask_code, dmd_mask, dmd_mask_ratio = get_mask_code(pred_code, mode=ratio_mode, value=mask_value, codebook_size=codebook_size)
                    # DDMD loss
                    with torch.no_grad():
                        # calculate the teacher and auxiliary model logits
                        true_cfg_ = (1 + (true_cfg-1) * dmd_mask_ratio) if adaptive_cfg else true_cfg
                        pred_logit_true = cfg_pred_model(teacher_model, dmd_mask_code, encoder_hidden_states, cond_embeds, \
                                                        img_ids, txt_ids, micro_conds, mask_prob=dmd_mask_ratio, \
                                                        uncond_encoder_hidden_states=neg_encoder_hidden_states, uncond_cond_embeds=neg_cond_embeds, \
                                                        codebook_size=codebook_size, cfg=true_cfg_)
                        pred_logit_fake = cfg_pred_model(fake_model, dmd_mask_code, encoder_hidden_states, cond_embeds, \
                                                        img_ids, txt_ids, micro_conds, mask_prob=dmd_mask_ratio, \
                                                        uncond_encoder_hidden_states=uncond_encoder_hidden_states, uncond_cond_embeds=uncond_cond_embeds, \
                                                        codebook_size=codebook_size, cfg=fake_cfg_eval)
                        # calculate the prob and log prob
                        pred_prob_true = torch.softmax(pred_logit_true / true_temp, -1)
                        pred_prob_fake = torch.softmax(pred_logit_fake / fake_temp, -1)
                        pred_logit, pred_prob_true, pred_prob_fake, _ = filter_nan_mask(pred_logit, pred_prob_true, pred_prob_fake)
                        if distil_loss_type == 'RKL' or 'Jeffreys' in distil_loss_type:
                            log_pred_prob_true = torch.log_softmax(pred_logit_true / true_temp, dim=-1)
                            log_pred_prob_fake = torch.log_softmax(pred_logit_fake / fake_temp, dim=-1)
                            pred_logit, log_pred_prob_true, log_pred_prob_fake, _ = filter_nan_mask(pred_logit, log_pred_prob_true, log_pred_prob_fake)
                    # weight
                    with torch.no_grad():
                        # DEBUG: weight factor should be 1 at first, can be tuned later
                        # pred_prob = torch.softmax(pred_logit / gen_temp, -1)      # [B, P^2, V]
                        # weight_factor = abs(pred_prob.float() - pred_prob_true.float())
                        # weight_factor = weight_factor.mean(dim=[1,2], keepdim=True) if loss_reduction == "mean" else weight_factor.sum(dim=[1,2], keepdim=True)
                        weight_factor = 1.
                    # distillation gradient
                    if distil_loss_type == 'FKL':       # FKL: forward KL divergence --> mode covering
                        # FKL = sum_0^V (p_true * log(p_true / p_fake))     # FKL for each token: shape [B, P^2, 1]
                        # grad_FKL = (p_fake - p_true)                      # gradient of FKL w.r.t each possible token pred in V: shape [B, P^2, V]
                        grad = (pred_prob_fake - pred_prob_true)
                    elif distil_loss_type == 'RKL':     # RKL: reverse KL divergence --> mode seeking
                        # RKL = sum_0^V (p_fake * log(p_fake / p_true))     # RKL for each token: shape [B, P^2, 1]
                        # grad_RKL = p_fake * (log(p_fake / p_true) - RKL) # gradient of RKL w.r.t each possible token pred in V: shape [B, P^2, V]
                        RKL = F.kl_div(log_pred_prob_true, pred_prob_fake, reduction='none').sum(dim=-1)    # sum over V: shape [B, P^2]
                        coef = log_pred_prob_fake - log_pred_prob_true - RKL.unsqueeze(-1)
                        grad = coef * pred_prob_fake
                    elif 'Jeffreys' in distil_loss_type:    # mixtures of FKL and RKL
                        grad_FKL = (pred_prob_fake - pred_prob_true)
                        RKL = F.kl_div(log_pred_prob_true, pred_prob_fake, reduction='none').sum(dim=-1)
                        coef = log_pred_prob_fake - log_pred_prob_true - RKL.unsqueeze(-1)
                        grad_RKL = coef * pred_prob_fake
                        grad = (1-Jeffreys_beta) * grad_FKL + Jeffreys_beta * grad_RKL
                    # consider the unmasked tokens only
                    grad = torch.where(dmd_mask.view(bsz,-1).unsqueeze(-1).to(grad.device), grad, 0.)
                    grad = grad.detach() / weight_factor
                    grad = torch.nan_to_num(grad)
                    loss_dm = 0.5 * F.mse_loss(pred_logit.float(), (pred_logit-grad).detach().float(), reduction=loss_reduction)
                    loss_dm = loss_dm / train_batch_size if loss_reduction == "sum" else loss_dm
                    loss_dm = loss_dm / gradient_accumulation_steps
                    loss_dict['loss_dm'] += accelerator.gather(loss_dm.repeat(train_batch_size)).mean().item()
                    loss_dict['score_diff'] += accelerator.gather((pred_prob_fake - pred_prob_true).detach().mean().repeat(train_batch_size)).mean().item()
                    generator_loss = dm_loss_weight * loss_dm
                else:
                    generator_loss = torch.tensor(0.0, device=accelerator.device)
                
                loss_dict["loss_generator"] += accelerator.gather(generator_loss.repeat(train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(generator_loss)

            total_norm = 0.0
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    total_norm += grad_norm.item() ** 2
                    # accelerator.log({'gradients/'+name: grad_norm.item()}, global_step)
            total_norm = total_norm ** 0.5
            accelerator.log({'total_gen_grad_norm': total_norm}, global_step)
            # generator.eval().requires_grad_(False)
            if '16' in mixed_precision:
                accelerator.clip_grad_norm_(generator.parameters(), max_grad_norm) if max_grad_norm > 0 else None
            # Update generator
            for param in generator.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer_g.step()
            optimizer_g.zero_grad(set_to_none=True)
        lr_scheduler_g.step()

        #################################################################################
        #                               update fake_model                               #
        #################################################################################
        # zero out the gradient of fake_model --> ensures clean gradient computation
        if dm_loss_weight > 0:
            # fake_model.train().requires_grad_(True)
            optimizer_fake.zero_grad(set_to_none=True)
        gradient_accumulation_steps_fake = 0 if (dm_loss_weight == 0) else gradient_accumulation_steps
        for accu_iter in range(gradient_accumulation_steps_fake):
            # # get the batch
            batch = next(batch_gen)
            with torch.no_grad():
                encoder_hidden_states, cond_embeds = compute_embeddings(text_encoder, tokenizer, batch["input_ids"], text_encoder_architecture, accelerator.device)
                pred_logit, pred_code = generate_one_step(generator, encoder_hidden_states, cond_embeds, 
                                                fixed_ratio=fixed_ratio, device=accelerator.device,
                                                micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                                patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value,
                                                top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=cur_noise_emb_perturb)
                pred_prob = torch.softmax(pred_logit / temperature_fake, -1)      # [B, P^2, V]

            # Drop xx% of the condition for cfg
            if fake_cfg_train != 1.:
                # replace batch condition with uncond batch with probability fake_cfg_drop_ratio
                assert encoder_hidden_states is not None
                drop_label = torch.empty((bsz, 1, 1), device=accelerator.device).float().uniform_(0, 1) < fake_cfg_drop_ratio
                encoder_hidden_states = torch.where(drop_label, uncond_encoder_hidden_states, encoder_hidden_states)
                cond_embeds = torch.where(drop_label.squeeze(-1), uncond_cond_embeds, cond_embeds)
            
            # Mask the encoded tokens
            masked_code, fake_mask, fake_mask_ratio = get_mask_code(pred_code, mode=ratio_mode_fake, value=mask_value, codebook_size=codebook_size)
            pred_logit_fake = cfg_pred_model(fake_model, masked_code, encoder_hidden_states, cond_embeds, \
                                            img_ids, txt_ids, micro_conds, mask_prob=fake_mask_ratio, \
                                            uncond_encoder_hidden_states=uncond_encoder_hidden_states, uncond_cond_embeds=uncond_cond_embeds, \
                                            codebook_size=codebook_size, cfg=fake_cfg_train)

            pred_logit_fake, pred_logit, _, _ = filter_nan_mask(pred_logit_fake, pred_logit)

            # KL divergence for soft targets: pred_prob
            if alpha_fake > 0:
                soft_pred_prob_fake = torch.log_softmax(pred_logit_fake / temperature_fake, dim=-1)
                loss_fake_soft = F.kl_div(soft_pred_prob_fake, pred_prob, reduction='none') * (temperature_fake ** 2)
                loss_fake_soft = loss_fake_soft.mean(dim=-1)
                loss_fake_soft = loss_fake_soft.view(-1, patch_size, patch_size)
                # loss_fake_soft = loss_fake_soft * mask.to(loss_fake_soft.device)
                loss_fake_soft = torch.where(fake_mask.to(loss_fake_soft.device), loss_fake_soft, 0.)
                loss_fake_soft = loss_fake_soft.sum() / fake_mask.sum()
            else:
                loss_fake_soft = torch.tensor(0.0, device=accelerator.device)
            # Cross-entropy loss for hard targets: pred_code --> only calculate the loss for the masked tokens
            pred_code = torch.where(fake_mask.to(pred_code.device), pred_code, ignore_index)
            loss_fake_hard = F.cross_entropy(pred_logit_fake.reshape(-1, codebook_size), pred_code.view(-1), ignore_index=ignore_index, label_smoothing=0.1, reduction='mean')
            loss_fake = alpha_fake * loss_fake_soft + (1 - alpha_fake) * loss_fake_hard
            loss_fake = loss_fake / gradient_accumulation_steps
            loss_dict['loss_fake'] += (accelerator.gather(loss_fake.repeat(train_batch_size)).mean().item() / fake_rounds)
            # Backpropagate
            accelerator.backward(loss_fake)

        if dm_loss_weight > 0:
            # fake_model.eval().requires_grad_(False)
            # accelerator.clip_grad_norm_(fake_model.parameters(), max_grad_norm)
            # Update fake score network
            for param in fake_model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer_fake.step()
            optimizer_fake.zero_grad(set_to_none=True)
            lr_scheduler_fake.step()
        
        ########################## post training step ##########################
        if COMPUTE_GENERATOR_GRADIENT:  # only when generator is updated
            if ema_decay > 0 and global_step % ema_freq == 0:
                update_ema(ema_model.parameters(), generator.parameters(), ema_decay, global_step)
            
            # log the losses and misc
            # logs = {key: loss_dict[key] for key in loss_dict}
            logs = {key: value for key, value in loss_dict.items() if value != 0}
            logs["lr_g"] = lr_scheduler_g.get_last_lr()[0]
            if dm_loss_weight > 0:
                logs["lr_fake"] = lr_scheduler_fake.get_last_lr()[0]
            logs["emb_perturb"] = cur_noise_emb_perturb
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            # reset loss_dict for next step (all grads are accumulated and step is done)
            for key in loss_dict:
                loss_dict[key] = 0.0

            # Save the model checkpoint periodically
            if accelerator.is_main_process and global_step > 0 and (global_step % checkpointing_steps == 0 or global_step in checkpointing_steps_tuple):
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if checkpoints_total_limit is not None:
                    checkpoints = os.listdir(output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)
                # save the state
                if global_step in checkpointing_steps_tuple:
                    save_path = os.path.join(output_dir, f"_checkpoint-{global_step}")
                else:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                save_all_state(os.path.join(output_dir, f"meta_checkpoints"), save_accelerator_state=True, global_step=global_step)
                save_all_state(save_path)
                logger.info(f"Saved state to {save_path}")

            # Periodically validation
            if accelerator.is_main_process and validation_steps and global_step > 0 and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                # logger current step and all logs
                logger.info(f"---> step {global_step}, logs: {logs}")
                log_validation(val_pipeline, generator, vqvae, accelerator, global_step, infer_step=1, fixed_ratio=fixed_ratio, patch_size=patch_size, local_seed=global_seed, codebook_size=codebook_size, mask_value=mask_value, micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids, name="generator", log_dir=f"{output_dir}/samples_val", noise_emb_perturb=cur_noise_emb_perturb)
                # log_validation(val_pipeline, teacher_model, vqvae, accelerator, global_step, infer_step=sampling_step, local_seed=global_seed, name="teacher_model", log_dir=f"{output_dir}/samples_val")
                if ema_decay > 0:
                    log_validation(val_pipeline, ema_model.to(accelerator.device), vqvae, accelerator, global_step, infer_step=1, fixed_ratio=fixed_ratio, patch_size=patch_size, local_seed=global_seed, codebook_size=codebook_size, mask_value=mask_value, micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids, name="ema_model", log_dir=f"{output_dir}/samples_val", noise_emb_perturb=cur_noise_emb_perturb)
                    ema_model.to('cpu' if ema_cpu else accelerator.device)
                if dm_loss_weight > 0:
                    log_validation(val_pipeline, fake_model, vqvae, accelerator, global_step, infer_step=sampling_step, local_seed=global_seed, name="fake_model", log_dir=f"{output_dir}/samples_val")
            
            progress_bar.update(1)
            global_step += 1
            
        accelerator.wait_for_everyone()

        if global_step >= max_train_steps:
            break
            
    accelerator.end_training()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    
    known_args, unknown_args = parser.parse_known_args()
    print('unknown_args:', unknown_args)
    
    # Load the config file
    config = OmegaConf.load(known_args.config)
    
    # Add arguments for each key in the config
    for key in config.keys():
        parser.add_argument(f"--{key}", type=type(config[key]))
    
    # Parse all arguments again
    args = parser.parse_args()
    
    # Update config with command-line arguments
    for key, value in vars(args).items():
        if key in config and value is not None:
            config[key] = value
            print(f"Updating config: {key} -> {value}")
    
    # name = Path(args.config).stem
    name = ''
    main(name=name, **config)
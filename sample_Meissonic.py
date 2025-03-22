import os
import re
import random
import numpy as np
import torch
import tqdm
import argparse
from torchvision.utils import save_image
import PIL.Image
from Meissonic_src.transformer import Transformer2DModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import VQModel
from Meissonic_src.dataset_utils import compute_embeddings

from omegaconf import OmegaConf
from pathlib import Path


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


#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.

    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.

    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

def init():
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    init_multiprocessing(rank=get_rank(), sync_device=sync_device)

def read_file_to_sentences(filename):
    # Initialize an empty list to store the sentences
    sentences = []
    # Open the file
    with open(filename, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in file:
            # Strip newline and any trailing whitespace characters
            clean_line = line.strip()
            # Add the cleaned line to the list if it is not empty
            if clean_line:
                sentences.append(clean_line)
    return sentences

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges



parser = argparse.ArgumentParser()
parser.add_argument("--save_dir",
                    type=str,
                    help="location of fake images for evaluation")
parser.add_argument("--guidance_scale",
                    type=float,
                    default=1.5,
                    help="guidance scale")
parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="random seed. -1 means not fixing random seed.")
parser.add_argument("--vit_path",
                    type=str,
                    default=None,
                    help="path to the pretrained vit model")
parser.add_argument("--mode",
                    type=str,
                    default='generation',
                    help="path to the pretrained vit model")
parser.add_argument("--nb_sample",
                    type=int,
                    default=50000,
                    help="number of samples")
parser.add_argument("--gen_temp",
                    type=float,
                    default=1.0,
                    help="temperature for sampling")
parser.add_argument("--noise_emb_perturb",
                    type=float,
                    default=0.1,
                    help="noise embedding perturbation")
parser.add_argument("--fixed_ratio",
                    type=float,
                    default=0.6,
                    help="fixed ratio of masked tokens")
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

init()

noise_emb_perturb = args.noise_emb_perturb
fixed_ratio = args.fixed_ratio

# load models
pretrained_model_name_or_path = 'meissonflow/meissonic'
text_encoder_architecture = "open_clip"
resolution = 1024
num_images_per_prompt = 2

# load vae
vae = VQModel.from_pretrained(pretrained_model_name_or_path, subfolder="vqvae")
vae = vae.eval()
vae = vae.to(device)
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
patch_size = resolution // vae_scale_factor     # Load VQGAN patch size
model = Transformer2DModel.from_pretrained(args.vit_path, subfolder="transformer")
model = model.eval()
model = model.to(device)
# Load text encoder
if text_encoder_architecture == "open_clip":
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    # text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")    # DEBUG: using original text enc for stable sampling
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
else:
    raise ValueError(f"Unknown text encoder architecture: {text_encoder_architecture}")
text_encoder = text_encoder.eval()
text_encoder = text_encoder.to(device)

codebook_size = model.config.codebook_size
mask_value = model.config.vocab_size - 1


# local_seed = args.seed
# setup_seed(args.seed)
# os.makedirs(args.save_dir, exist_ok=True)

# Create uncond embeds for classifier free guidance
# uncond_prompt_embeds = torch.zeros(train_batch_size, MAX_SEQ_LENGTH, 2048).to(accelerator.device)
with torch.no_grad():
    empty_embeds, empty_clip_embeds = compute_embeddings(text_encoder, tokenizer, "", text_encoder_architecture, device)
    micro_conds = torch.tensor([1024, 1024, 0, 0, 6.0])     # [orig_width, orig_height, c_top, c_left, hps_score]
    micro_conds = micro_conds.unsqueeze(0).expand(16, -1).to(device)
    micro_conds = micro_conds[0].unsqueeze(0).repeat(num_images_per_prompt, 1)
    if resolution == 1024: # only stage 3 and stage 4 do not apply 2*
        img_ids = _prepare_latent_image_ids(16, patch_size, patch_size, device)
    else:
        img_ids = _prepare_latent_image_ids(16, 2*patch_size, 2*patch_size, device)
    txt_ids = torch.zeros(empty_embeds.shape[1], 3).to(device=device)


# create save directory
os.makedirs(args.save_dir, exist_ok=True)
print("saving into", args.save_dir)

top_k = 0
top_p = 0.

if args.mode == 'sample':
    with open('./hand_pick_prompt.txt', "r") as f:
        prompts = f.readlines()
    # for gen_temp in [0.1, 0.5, 1.0, 1.5, 2, 5]:
    print(f"Generating images at fixed_r {fixed_ratio} and noise_pert {noise_emb_perturb}")
    gen_temp_list = [1.]
    top_k_list = [top_k]
    top_p_list = [top_p]
    local_seed = args.seed
    for gen_temp in gen_temp_list:
        for top_k in top_k_list:
            for top_p in top_p_list:
                print(f"current temperature {gen_temp}, top_k {top_k}, top_p {top_p}")
                setup_seed(local_seed)
                for index, prompt in enumerate(prompts):
                    # Get the text embedding
                    empty_embeds, empty_clip_embeds = compute_embeddings(text_encoder, tokenizer, prompt, device=device)
                    encoder_hidden_states = empty_embeds.repeat(num_images_per_prompt, 1, 1)
                    cond_embeds = empty_clip_embeds.repeat(num_images_per_prompt, 1)
                    images = sample_one_step(model, vae, encoder_hidden_states=encoder_hidden_states, cond_embeds=cond_embeds, fixed_ratio=fixed_ratio,
                                                    patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                                                    micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                                    top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=noise_emb_perturb)
                    images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                    for idx, image_np_ in enumerate(images_np):
                        image_name = f"{index}_{prompt.replace(' ','_').replace('/','')}"
                        image_name = image_name.replace('\n','')
                        image_path = f"{args.save_dir}/{image_name}_{idx:06d}.png"
                        image_path = f"{args.save_dir}/{index}_{idx:06d}.png"
                        image = PIL.Image.fromarray(image_np_.numpy()).save(image_path)

            # network_dir='./models/meissonic'
            # CUDA_VISIBLE_DEVICES=0 torchrun \
            #         --nproc_per_node=1 \
            #         --master_port=29501 \
            #     sample_Meissonic.py \
            #     --save_dir ./samples/Meissonic_sample \
            #     --vit_path $network_dir \
            #     --mode sample \
                
elif args.mode == 'generate_FID':
    from T2IBenchmark import calculate_fid
    from T2IBenchmark.datasets import get_coco_fid_stats
    from T2IBenchmark.datasets import get_coco_30k_captions
    # text_prompts = '/home/polytechnique/x-yuanzhi.zhu/codes/gigagan_coco_captions.txt'
    # captions = read_file_to_sentences(text_prompts)
    # get COCO-30k captions
    id2caption = get_coco_30k_captions()
    captions = []
    ids = []
    for d in id2caption.items():
        ids.append(d[0])
        captions.append(d[1])
    
    max_batch_size = 16
    seeds = list(range(30000))
    num_batches = ((len(seeds) - 1) // (max_batch_size * get_world_size()) + 1) * get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[get_rank() :: get_world_size()]
    num_images_per_prompt = 1
    micro_conds = micro_conds[0].unsqueeze(0).repeat(max_batch_size, 1)
    
    # Set the local random seed for reproducibility
    local_seed = args.seed + get_rank()
    setup_seed(local_seed)

    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(get_rank() != 0)):
        torch.distributed.barrier()
        prompts = [captions[i] for i in batch_seeds.tolist()]  # Index captions using list comprehension
        batched_generation = True
        num_images = len(prompts) if batched_generation else 1

        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        encoder_hidden_states, cond_embeds = compute_embeddings(text_encoder, tokenizer, prompts, device=device)
        images = sample_one_step(model, vae, encoder_hidden_states=encoder_hidden_states, cond_embeds=cond_embeds, fixed_ratio=fixed_ratio,
                                        patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                                        micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                        top_k=top_k, top_p=top_p, gen_temp=args.gen_temp, noise_emb_perturb=noise_emb_perturb)
        images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
        
        for i, prompt in enumerate(prompts):
            sanitized_prompt = prompt.replace(" ", "_")
            file_path = os.path.join(args.save_dir, f"{batch_seeds[i]}_{sanitized_prompt}_{resolution}.png")
            image = PIL.Image.fromarray(images_np[i].numpy()).save(file_path)
        
    fid, _ = calculate_fid(
        args.save_dir,
        get_coco_fid_stats()
    )
    print(f'FID: {fid} for {args.save_dir}')

            # network_dir='./models/meissonic'
            # CUDA_VISIBLE_DEVICES=0 torchrun \
            #         --nproc_per_node=1 \
            #         --master_port=29501 \
            #     sample_Meissonic.py \
            #     --save_dir ./samples/Meissonic_coco \
            #     --vit_path $network_dir \
            #     --mode generate_FID \
            #     --gen_temp 1.2 \

elif args.mode == 'generate_HPSV2':
    import os
    import hpsv2

    # pip install hpsv2
    micro_conds = micro_conds[0].unsqueeze(0).repeat(1, 1)
    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all')

    # Set the local random seed for reproducibility
    local_seed = args.seed + get_rank()
    setup_seed(local_seed)

    # Iterate over the benchmark prompts to generate images
    for style, prompts in tqdm.tqdm(all_prompts.items(), unit='style'):
        print(f'Generating images for style: {style}, total prompt number: {len(prompts)}')
        for idx, prompt in tqdm.tqdm(enumerate(prompts), unit='prompt', leave=False):
            encoder_hidden_states, cond_embeds = compute_embeddings(text_encoder, tokenizer, prompt, device=device)
            images = sample_one_step(model, vae, encoder_hidden_states=encoder_hidden_states, cond_embeds=cond_embeds, fixed_ratio=fixed_ratio,
                                            patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                                            micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                            top_k=top_k, top_p=top_p, gen_temp=args.gen_temp, noise_emb_perturb=noise_emb_perturb)
            images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
            
            file_path = os.path.join(args.save_dir, style, f"{idx:05d}.jpg")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            image = PIL.Image.fromarray(images_np[0].numpy()).save(file_path)
            # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().
    
    hpsv2.evaluate(args.save_dir, hps_version="v2.0") 

            # network_dir='./models/meissonic'
            # CUDA_VISIBLE_DEVICES=0 torchrun \
            #         --nproc_per_node=1 \
            #         --master_port=29501 \
            #     sample_Meissonic.py \
            #     --save_dir ./samples/Meissonic_hps \
            #     --vit_path $network_dir \
            #     --mode generate_HPSV2 \
            #     --gen_temp 1.2 \

elif args.mode == 'generate_GenEval':
    # pip install mmcv-full==1.7.1
    batch_size = 8
    n_samples = 4
    import json
    micro_conds = micro_conds[0].unsqueeze(0).repeat(1, 1)
    # git clone https://github.com/open-mmlab/mmdetection.git
    # cd mmdetection; git checkout 2.x
    # pip install -v -e .
    # wget https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl -O samples/evaluation_metadata.jsonl

    metadata_file = 'samples/evaluation_metadata.jsonl'
    with open(metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    print(f"Generating images for {len(metadatas)} prompts")
    for index, metadata in enumerate(metadatas):
        # Set the local random seed for reproducibility
        local_seed = args.seed + get_rank()
        setup_seed(local_seed)

        outpath = os.path.join(args.save_dir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        with torch.no_grad():
            all_samples = list()
            for n in tqdm.trange((n_samples)):
                # Generate images
                encoder_hidden_states, cond_embeds = compute_embeddings(text_encoder, tokenizer, prompt, device=device)
                images = sample_one_step(model, vae, encoder_hidden_states=encoder_hidden_states, cond_embeds=cond_embeds, fixed_ratio=fixed_ratio,
                                                patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                                                micro_conds=micro_conds, img_ids=img_ids, txt_ids=txt_ids,
                                                top_k=top_k, top_p=top_p, gen_temp=args.gen_temp, noise_emb_perturb=noise_emb_perturb)
                images_np = (images * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                
                file_path = os.path.join(sample_path, f"{sample_count:05}.png")
                image = PIL.Image.fromarray(images_np[0].numpy()).save(file_path)
                sample_count += 1
        
            # network_dir='./models/meissonic'
            # CUDA_VISIBLE_DEVICES=0 torchrun \
            #         --nproc_per_node=1 \
            #         --master_port=29501 \
            #     sample_Meissonic.py \
            #     --save_dir ./samples/Meissonic_GenEval \
            #     --vit_path $network_dir \
            #     --mode generate_GenEval \
            #     --gen_temp 1.2 \

    # git clone https://github.com/open-mmlab/mmdetection.git
    # cd mmdetection; git checkout 2.x
    # pip install -v -e .
    # wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "samples_test/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    
    # python evaluation/evaluate_images.py \
    #     "<IMAGE_FOLDER>" \
    #     --outfile "samples_test/GenEval_results.jsonl" \
    #     --model-path "samples_test/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"



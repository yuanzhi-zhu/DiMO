import os
import sys
sys.path.append("./")
import numpy as np
import random
import re
import PIL
import tqdm

import torch
from torchvision import transforms
from Meissonic_src.transformer import Transformer2DModel
from Meissonic_src.pipeline import Pipeline
from Meissonic_src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats
from T2IBenchmark.datasets import get_coco_30k_captions

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


init()

seed = 1024
device = 'cuda'

model_path = "MeissonFlow/Meissonic"
model = Transformer2DModel.from_pretrained(model_path,subfolder="transformer",)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
# text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer",)
scheduler = Scheduler.from_pretrained(model_path,subfolder="scheduler",)
pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to(device)

# text_prompts = '/home/polytechnique/x-yuanzhi.zhu/codes/gigagan_coco_captions.txt'
# captions = read_file_to_sentences(text_prompts)
# get COCO-30k captions
id2caption = get_coco_30k_captions()
captions = []
ids = []
for d in id2caption.items():
    ids.append(d[0])
    captions.append(d[1])

max_batch_size = 8
seeds = list(range(len(captions)))
print(f'using {get_world_size()} GPUs')
num_batches = ((len(seeds) - 1) // (max_batch_size * get_world_size()) + 1) * get_world_size()
all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
rank_batches = all_batches[get_rank() :: get_world_size()]

# rnd = StackedRandomGenerator(device, batch_seeds)
local_seed = seed + get_rank()
print(f"Local seed: {local_seed}, current rank: {get_rank()}")
# Set the local random seed for reproducibility
torch.manual_seed(local_seed)
torch.cuda.manual_seed_all(local_seed)
np.random.seed(local_seed)
random.seed(local_seed)
torch.backends.cudnn.deterministic = True

steps = 64
CFG = 9
resolution = 1024
negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
output_dir = "./samples"
os.makedirs(output_dir, exist_ok=True)

for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(get_rank() != 0)):
    # import pdb; pdb.set_trace()
    prompts = [captions[i] for i in batch_seeds.tolist()]
    batched_generation = True
    num_images = len(prompts) if batched_generation else 1

    images = pipe(
        prompt=prompts[:num_images], 
        negative_prompt=[negative_prompt] * num_images,
        height=resolution,
        width=resolution,
        guidance_scale=CFG,
        num_inference_steps=steps
        ).images

    for i, prompt in enumerate(prompts[:num_images]):
        sanitized_prompt = prompt.replace(" ", "_")
        file_path = os.path.join(output_dir, f"{batch_seeds[i]}_{sanitized_prompt}_{resolution}_{steps}_{CFG}.png")
        images[i].save(file_path)
        # print(f"The {i+1}/{num_images} image is saved to {file_path}")

fid, _ = calculate_fid(
    output_dir,
    get_coco_fid_stats()
)
print(f'FID: {fid} for {output_dir}')


# export PYTHONPATH=$PYTHONPATH:../
# CUDA_VISIBLE_DEVICES=0,1 python sample.py
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sample.py
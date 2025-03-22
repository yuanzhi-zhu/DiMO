import os
import random
import numpy as np
import torch
import tqdm
import argparse
from torchvision.utils import save_image
import PIL.Image

from MaskGit_src.Network.transformer import MaskTransformer
from MaskGit_src.Network.Taming.models.vqgan import VQModel
from omegaconf import OmegaConf
from pathlib import Path


def generate_one_step(model, nb_sample=128, y=None, fixed_ratio=0.5, device='cuda',
                        patch_size=16, codebook_size=1024, mask_value=1024,
                        top_k=0, top_p=0., gen_temp=1., noise_emb_perturb=0.,):
    bsz = nb_sample
    # 1. initial random code and conditions
    if y is None:
        y = torch.randint(0, 1000, (bsz,)).to(device, dtype=torch.long)
        # y = torch.ones((bsz,)).to(device, dtype=torch.long) * 282
    assert y.size(0) == bsz
    shape_code = torch.ones(bsz, patch_size, patch_size)
    code = torch.randint_like(shape_code, 0, codebook_size).to(device, dtype=torch.long)
    # 2. add mask to fixed ratio
    masked_code = code.detach().clone()
    # Sample the amount of tokens + localization to mask
    num_token_masked = torch.tensor(patch_size**2 * fixed_ratio).round().clamp(min=1)
    batch_randperm = torch.rand(bsz, patch_size**2).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask = mask.reshape(bsz, patch_size, patch_size)
    masked_code[mask] = torch.full_like(masked_code[mask], mask_value)
    # 3. one-step generator prediction
    pred_logit = model(masked_code, y, noise_emb_perturb=noise_emb_perturb)  # The unmasked tokens prediction: logits: [B, P^2, V]
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

    # 4. Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)      # [B, P^2, V]
    pred_distri = torch.distributions.Categorical(probs=probs)
    pred_code = pred_distri.sample()
    pred_code = pred_code.view(bsz, patch_size, patch_size)     # [B, P, P]

    return pred_logit, pred_code, y


@torch.no_grad()
def sample_one_step(model, vae, nb_sample=10, labels=None, fixed_ratio=0.5,
                        patch_size=16, codebook_size=1024, mask_value=1024, device='cuda',
                        top_k=0, top_p=0., gen_temp=1., noise_emb_perturb=0.):
    model.eval()
    if labels is None:  # Default classes generated
        # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
        # # all tiger cat
        # labels = [282] * (nb_sample)
        labels = labels + [random.randint(0, 999) for _ in range(nb_sample - len(labels))]
        labels = torch.LongTensor(labels).to(device)
    _, code, _ = generate_one_step(model, nb_sample, labels, fixed_ratio=fixed_ratio,
                        patch_size=patch_size, codebook_size=codebook_size, mask_value=mask_value, device=device,
                        top_k=top_k, top_p=top_p, gen_temp=gen_temp, noise_emb_perturb=noise_emb_perturb)
    # decode the final prediction
    _code = torch.clamp(code, 0,  codebook_size-1)
    x = vae.decode_code(_code)
    model.train()
    return x


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

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


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

if args.mode == 'generate':
    print(f'mode: {args.mode}, nb_sample: {args.nb_sample}')
    init()

noise_emb_perturb = args.noise_emb_perturb
fixed_ratio = args.fixed_ratio

# load models
model = MaskTransformer.from_pretrained(args.vit_path, 
        img_size=256, hidden_dim=768, codebook_size=1024, depth=24, heads=16, mlp_dim=3072, dropout=0.1)
model = model.eval()
model = model.to('cuda')
# load vae
network_path = "./pretrained_maskgit/VQGAN/"
config = OmegaConf.load(network_path + "model.yaml")
vae = VQModel(**config.model.params)
checkpoint = torch.load(network_path + "last.ckpt", map_location="cpu")["state_dict"]
vae.load_state_dict(checkpoint, strict=False)
vae = vae.eval()
vae = vae.to('cuda')

# create save directory
os.makedirs(args.save_dir, exist_ok=True)
print("saving into", args.save_dir)


if args.mode == 'sample':
    local_seed = args.seed
    nb_sample = args.nb_sample
    gen_temp_list = [1.]
    top_k_list = [0]
    top_p_list = [0.]
    print(f"Sample {args.nb_sample} images at fix_r {fixed_ratio} and noise_pert {noise_emb_perturb} with temperature {args.gen_temp}")
    Path(f"{args.save_dir}").mkdir(parents=True, exist_ok=True)
    for gen_temp in gen_temp_list:
        for top_k in top_k_list:
            for top_p in top_p_list:
                # set random seed
                setup_seed(local_seed)
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 999, random.randint(0, 999)] * (nb_sample // 10)
                labels = labels + [random.randint(0, 999) for _ in range(nb_sample - len(labels))]
                # labels = torch.randint(0, 1000, (nb_sample,))
                labels = torch.LongTensor(labels).to('cuda')
                gen_sample = sample_one_step(model, vae, nb_sample=nb_sample, labels=labels, 
                                            fixed_ratio=fixed_ratio, noise_emb_perturb=noise_emb_perturb, 
                                            patch_size=16, codebook_size=1024, mask_value=1024, 
                                            gen_temp=gen_temp, 
                                            top_k=top_k, top_p=top_p, device='cuda')
                Path(f"{args.save_dir}/seed{local_seed}").mkdir(parents=True, exist_ok=True)
                save_path = f"{args.save_dir}/seed{local_seed}/fixed_ratio_{fixed_ratio}_noise_emb_perturb_{noise_emb_perturb}_gen_temp{gen_temp}-topk{top_k}-topp{top_p}.png"
                save_image(gen_sample/2+0.5, save_path, nrow=8, padding=2, normalize=False)

            # network_dir="./models/maskgit"
            # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample_MaskGit.py \
            #     --save_dir ./samples/test_sample \
            #     --vit_path $network_dir \
            #     --mode sample \
            #     --nb_sample 64

elif args.mode == 'generate':
    print(f"Generating {args.nb_sample} images at fix_r {fixed_ratio} and noise_pert {noise_emb_perturb} with temperature {args.gen_temp}")
    print(f'using number of GPUs: {get_world_size()}, rank: {get_rank()}')

    # prepare all classes
    nb_sample = args.nb_sample
    all_labels = torch.arange(1000).to(dtype=torch.long)
    all_labels = all_labels.repeat((nb_sample // 1000) + 1)[:nb_sample]
    # all_labels = all_labels.repeat_interleave((nb_sample // 1000) + 1)[:nb_sample]

    # indices on each GPU
    batch_size = 16
    seeds = list(range(nb_sample))
    num_batches = ((len(seeds) - 1) // (batch_size * get_world_size()) + 1) * get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[get_rank() :: get_world_size()]
    
    local_seed = args.seed + get_rank()
    # Set the local random seed for reproducibility
    setup_seed(local_seed)
    
    print("start to generate")
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(get_rank() != 0)):
        torch.distributed.barrier()
        labels = all_labels[batch_seeds].to('cuda')
        gen_sample = sample_one_step(model, vae, nb_sample=labels.shape[0], labels=labels, 
                                    fixed_ratio=fixed_ratio, noise_emb_perturb=noise_emb_perturb, 
                                    patch_size=16, codebook_size=1024, mask_value=1024, 
                                    gen_temp=args.gen_temp, 
                                    top_k=0, top_p=0., device='cuda')
        
        images_np = (gen_sample * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i, image_np_ in enumerate(images_np):
            image_path = os.path.join(args.save_dir, f'{batch_seeds[i]:06d}.png')
            if image_np_.shape[2] == 1:
                PIL.Image.fromarray(image_np_[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np_, 'RGB').save(image_path)

            # network_dir="./models/maskgit"
            # CUDA_VISIBLE_DEVICES=$DID torchrun --nproc_per_node=$num_gpu_per_node sample_MaskGit.py \
            #     --save_dir ./samples/50000 \
            #     --vit_path $network_dir \
            #     --mode generate \
            #     --gen_temp 7. \
            #     --nb_sample 50000

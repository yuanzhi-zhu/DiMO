import os
import random
import PIL.Image
import logging

import numpy as np
import argparse

import torch
import torchvision.utils as vutils
import tqdm
import matplotlib.pyplot as plt

from Trainer.vit import MaskGIT


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='generate', help="Mode of the script")
parser.add_argument("--nb_sample", type=int, default=5000, help="Number of sample")
parser.add_argument("--save_dir", type=str, default="samples_5000", help="Directory to save the samples")
parser.add_argument("--vit_folder", type=str, default="../pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth", help="Path to the MaskGIT model")
parser.add_argument("--vqgan_folder", type=str, default="../pretrained_maskgit/VQGAN/", help="Path to the VQGAN model")
parser.add_argument("--log_file", type=str, default="fid_eval.log", help="Log file")
parser.add_argument("--mask_value", type=int, default=1024, help="Value of the masked token")
parser.add_argument("--img_size", type=int, default=256, help="Size of the image")
parser.add_argument("--path_size", type=int, default=16, help="Number of vizual token")
parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility")
parser.add_argument("--channel", type=int, default=3, help="Number of input channel")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
parser.add_argument("--data_folder", type=str, default="/datasets_local/ImageNet/", help="Data folder")
parser.add_argument("--writer_log", type=str, default="", help="Writer log")
parser.add_argument("--data", type=str, default="imagenet", help="Data")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
parser.add_argument("--iter", type=int, default=1_500_000, help="Number of iteration")
parser.add_argument("--global_epoch", type=int, default=380, help="Number of epoch")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--drop_label", type=float, default=0.1, help="Drop out label for cfg")
parser.add_argument("--resume", type=bool, default=True, help="Set to True for loading model")
parser.add_argument("--debug", type=bool, default=True, help="Load only the model (not the dataloader)")
parser.add_argument("--test_only", type=bool, default=False, help="Dont launch the testing")
parser.add_argument("--is_master", type=bool, default=True, help="Master machine")
parser.add_argument("--is_multi_gpus", type=bool, default=False, help="Set to False for colab demo")
parser.add_argument("--distil", type=bool, default=False, help="Distil")
parser.add_argument("--cfg", type=float, default=4.5, help="Classifier Free Guidance")
parser.add_argument("--step", type=int, default=8, help="Number of step")
parser.add_argument("--sm_temp", type=float, default=1.3, help="Softmax Temperature")
parser.add_argument("--r_temp", type=float, default=7., help="Gumbel Temperature")
parser.add_argument("--randomize", type=str, default="linear", help="Noise scheduler")
parser.add_argument("--sched_mode", type=str, default="arccos", help="Mode of the scheduler")
args = parser.parse_args()


# Fixe seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enable = False
    torch.backends.cudnn.deterministic = True

# Instantiate the MaskGIT
maskgit = MaskGIT(args)

def viz(x, nrow=10, pad=2, size=(18, 18), name=None):
    """
    Visualize a grid of images.

    Args:
        x (torch.Tensor): Input images to visualize.
        nrow (int): Number of images in each row of the grid.
        pad (int): Padding between the images in the grid.
        size (tuple): Size of the visualization figure.

    """
    nb_img = len(x)
    min_norm = x.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = x.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    x = (x - min_norm) / (max_norm - min_norm)

    x = vutils.make_grid(x.float().cpu(), nrow=nrow, padding=pad, normalize=False)
    plt.figure(figsize = size)
    plt.axis('off')
    plt.imshow(x.permute(1, 2, 0))
    # plt.show()
    if name:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)

def decoding_viz(gen_code, mask, maskgit, name=None):
    """
    Visualize the decoding process of generated images with associated masks.

    Args:
        gen_code (torch.Tensor): Generated code for decoding.
        mask (torch.Tensor): Mask used for decoding.
        maskgit (MaskGIT): MaskGIT instance.
    """
    start = torch.FloatTensor([1, 1, 1]).view(1, 3, 1, 1).expand(1, 3, maskgit.patch_size, maskgit.patch_size) * 0.8
    end = torch.FloatTensor([0.01953125, 0.30078125, 0.08203125]).view(1, 3, 1, 1).expand(1, 3, maskgit.patch_size, maskgit.patch_size) * 1.4
    code = torch.stack((gen_code), dim=0).squeeze()
    mask = torch.stack((mask), dim=0).view(-1, 1, maskgit.patch_size, maskgit.patch_size).cpu()

    with torch.no_grad():
        x = maskgit.ae.decode_code(torch.clamp(code, 0, 1023))

    binary_mask = mask * start + (1 - mask) * end
    binary_mask = vutils.make_grid(binary_mask, nrow=len(gen_code), padding=1, pad_value=0.4, normalize=False)
    binary_mask = binary_mask.permute(1, 2, 0)

    plt.figure(figsize = (18, 2))
    plt.gca().invert_yaxis()
    plt.pcolormesh(binary_mask, edgecolors='w', linewidth=.5)
    plt.axis('off')
    # plt.show()
    if name:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)

    viz(x, nrow=len(gen_code), name=f'{name}_decoded.png')

os.makedirs(args.save_dir, exist_ok=True)

sm_temp = args.sm_temp         # Softmax Temperature
r_temp = args.r_temp           # Gumbel Temperature
w = args.cfg                   # Classifier Free Guidance
randomize = args.randomize     # Noise scheduler
step = args.step               # Number of step
sched_mode = args.sched_mode   # Mode of the scheduler

if args.mode == 'sample':
    print(f"Sampling {args.nb_sample} images with temperature {sm_temp} and {r_temp}")
    # Sample 100 images with random labels
    labels, name = [1, 7, 282, 604, 724, 179, 681, 367, 850, random.randint(0, 999)] * 1, "r_row"
    labels = torch.LongTensor(labels).to(args.device)
    # Generate sample
    gen_sample, gen_code, l_mask = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, w=w, randomize=randomize, r_temp=r_temp, sched_mode=sched_mode, step=step)
    viz(gen_sample, nrow=10, size=(18, 18), name=f'{args.save_dir}/{randomize}sample_512_.png')

    labels, name = 7, "chicken"
    sample, code, mask = maskgit.sample(nb_sample=1, labels=torch.LongTensor([labels]).to(args.device), sm_temp=sm_temp, w=w, randomize=randomize, r_temp=r_temp, sched_mode=sched_mode, step=step)
    decoding_viz(code, mask, maskgit=maskgit, name=f'{args.save_dir}/{randomize}_sample_chicken_.png')
    viz(sample, size=(8,8), name=f'{args.save_dir}/{randomize}final_sample_chicken_.png')


elif args.mode == 'generate':
    gfile_stream = open(f'{args.log_file}', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logger.info(f"Generating {args.nb_sample} images with temperature {sm_temp} and {r_temp}")
    logger.info(f"cfg: {w}; randomize: {randomize}; sched_mode: {sched_mode}; step: {step}")
    # sample 50000 images with ranom labels
    batch_size = 16
    nb_sample = args.nb_sample
    num_batches = (nb_sample + batch_size - 1) // batch_size
    
    all_labels = torch.arange(1000).to(dtype=torch.long)
    all_labels = all_labels.repeat((nb_sample // 1000) + 1)[:nb_sample]
    all_labels = all_labels[torch.randperm(nb_sample)]
    img_idx = 0
    for i in tqdm.tqdm(range(num_batches)):
    # for i in (range(num_batches)):
        # labels = torch.randint(0, 1000, (batch_size,)).to('cuda', dtype=torch.long)
        labels = all_labels[i * batch_size: (i + 1) * batch_size].to('cuda')
        # Generate sample
        gen_sample, gen_code, l_mask = maskgit.sample(nb_sample=labels.size(0), labels=labels, sm_temp=sm_temp, w=w, randomize=randomize, r_temp=r_temp, sched_mode=sched_mode, step=step)
        images_np = (gen_sample * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for image_np_ in images_np:
            if img_idx >= nb_sample:
                break
            image_path = f"{args.save_dir}/{img_idx:06d}.png"
            if image_np_.shape[2] == 1:
                PIL.Image.fromarray(image_np_[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np_, 'RGB').save(image_path)
            img_idx += 1
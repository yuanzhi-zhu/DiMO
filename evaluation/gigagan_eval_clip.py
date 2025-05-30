# GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)
# See license file or visit https://github.com/mingukkang/GigaGAN for details

# evaluation.py


import os
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from gigagan_data_util import EvalDataset, CenterCropLongEdge
import torchvision.transforms as transforms
import logging
from torch.utils.data import Dataset
import imageio.v2 as imageio
from torchvision.utils import save_image

def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image



class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """ 
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class CLIPScoreDataset(Dataset):
    def __init__(self, images, captions, transform, preprocessor) -> None:
        super().__init__()
        self.images = images 
        self.captions = captions 
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        image = imageio.imread(image_path)
        image_pil = self.transform(image)
        image_pil = self.preprocessor(image_pil)
        caption = self.captions[index]
        return image_pil, caption 



def resize_and_center_crop(image_np, resize_size=256):
    image_pil = Image.fromarray(image_np) 
    image_pil = CenterCropLongEdge()(image_pil)

    if resize_size is not None:
        image_pil = image_pil.resize((resize_size, resize_size),
                                    Image.LANCZOS)
    return image_pil

def simple_collate(batch):
    images, captions = [], []
    for img, cap in batch:
        images.append(img)
        captions.append(cap)
    return images, captions


@torch.no_grad()
def compute_clip_score(
    images, captions, clip_model="ViT-B/32", device="cuda", how_many=30000, batch_size=64):
    print("Computing CLIP score")
    import clip as openai_clip 
    if clip_model == "ViT-B/32":
        clip, clip_preprocessor = openai_clip.load("ViT-B/32", device=device)
        clip = clip.eval()
    elif clip_model == "ViT-G/14":
        import open_clip
        clip, _, clip_preprocessor = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s12b_b42k")
        clip = clip.to(device)
        clip = clip.eval()
        clip = clip.float()
    else:
        raise NotImplementedError

    dataset = CLIPScoreDataset(
        images, captions, transform=resize_and_center_crop, 
        preprocessor=clip_preprocessor
    )
    logging.info(f'length of dataset: {len(dataset)}')
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False, num_workers=8,
        collate_fn=simple_collate
    )

    cos_sims = []
    count = 0
    # for imgs, txts in zip(images, captions):
    progress_bar = tqdm(total=len(dataloader), desc='Processing')
    for index, (imgs_pil, txts) in tqdm(enumerate(dataloader)):
        # imgs_pil = [resize_and_center_crop(imgs)]
        # txts = [txts]
        # imgs_pil = [clip_preprocessor(img) for img in imgs]
        imgs = torch.stack(imgs_pil, dim=0).to(device)

        # import pdb; pdb.set_trace()
        # save_image(imgs[:8]/2+0.5, 'test.png')

        tokens = openai_clip.tokenize(txts, truncate=True).to(device)
        # Prepending text prompts with "A photo depicts "
        # https://arxiv.org/abs/2104.08718
        prepend_text = "A photo depicts "
        prepend_text_token = openai_clip.tokenize(prepend_text)[:, 1:4].to(device)
        prepend_text_tokens = prepend_text_token.expand(tokens.shape[0], -1)
        
        start_tokens = tokens[:, :1]
        new_text_tokens = torch.cat(
            [start_tokens, prepend_text_tokens, tokens[:, 1:]], dim=1)[:, :77]
        last_cols = new_text_tokens[:, 77 - 1:77]
        last_cols[last_cols > 0] = 49407  # eot token
        new_text_tokens = torch.cat([new_text_tokens[:, :76], last_cols], dim=1)
        
        img_embs = clip.encode_image(imgs)
        text_embs = clip.encode_text(new_text_tokens)

        similarities = torch.nn.functional.cosine_similarity(img_embs, text_embs, dim=1)
        cos_sims.append(similarities)
        count += similarities.shape[0]

        progress_bar.update(1)
        # Optionally, you can set the progress bar description
        progress_bar.set_description(f"Processing {index+1}/{len(dataloader)}")

        if count >= how_many:
            break
    
    clip_score = torch.cat(cos_sims, dim=0)[:how_many].mean()
    clip_score = clip_score.detach().cpu().numpy()
    # import pdb; pdb.set_trace()
    return clip_score

def list_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']  # Add or remove extensions as needed
    image_files = []

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, filename))
    return image_files


def sort_image_paths(paths):
    def extract_number(path):
        return int(os.path.splitext(os.path.basename(path))[0].split('_step_')[0])
    
    return sorted(paths, key=extract_number)


def evaluate_model(opt):
    file_path = opt.caption_file

    # file_path = '/cpfs/data/user/yuazhu/codes/improved_perflow/scripts/fid/gigagan_coco_captions.txt'

    logging.info(f"caption file: {opt.caption_file}")
    logging.info(f"ref_dir: {opt.ref_dir}")
    
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())  # Add the line to the list, stripping newline characters

    images = list_images(opt.ref_dir)
    images = sort_image_paths(images)

    # import pdb;pdb.set_trace()
    clip_score = compute_clip_score(
        images=images[:opt.how_many],
        captions=lines[:opt.how_many],
        clip_model=opt.clip_model4eval,
        how_many=opt.how_many,
        batch_size=opt.batch_size,
    )
    logging.info(f"clip score ref_dir: {clip_score}")

    return


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--how_many", default=30000, type=int)
    parser.add_argument("--clip_model4eval", default="ViT-G/14", type=str, help="[WO, ViT-B/32, ViT-G/14]")

    parser.add_argument("--ref_dir",
                        default="",
                        help="location of the reference images for evaluation")
    parser.add_argument("--eval_res", default=256, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--caption_file",
                    type=str,
                    default='./prompts/captions.txt',
                    help="location of the file of captions")
    opt, _ = parser.parse_known_args()

    gfile_stream = open(f'clip_eval.log', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    evaluate_model(opt)
    logger.info(f'\n\n')


# CUDA_VISIBLE_DEVICES=2 python gigagan_eval_clip.py
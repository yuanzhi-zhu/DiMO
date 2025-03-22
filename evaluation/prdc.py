from cleanfid import fid
import torch
import os
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import argparse
import logging

from tqdm import tqdm

import torch
import numpy as np
import sklearn.metrics


def calculate_pr_dc(real_feats, fake_feats, data_loader, eval_model, num_generate, cfgs, quantize, nearest_k,
                    world_size, DDP, disable_tqdm):
    eval_model.eval()

    real_embeds = real_feats
    fake_embeds = np.array(fake_feats.detach().cpu().numpy(), dtype=np.float64)[:num_generate]

    metrics = compute_prdc(real_features=real_embeds, fake_features=fake_embeds, nearest_k=nearest_k)

    prc, rec, dns, cvg = metrics["precision"], metrics["recall"], metrics["density"], metrics["coverage"]
    return prc, rec, dns, cvg


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fid.build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)
        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


# https://github.com/openai/consistency_models_cifar10/blob/main/jcm/metrics.py#L117
def get_feature_np(
    samples,
    feat_model,
    batch_size=512,
    num_workers=12,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
    seed=0,
):
    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    l_feats = []
    for batch in tqdm(dataloader):
        l_feats.append(fid.get_batch_features(batch, feat_model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats

def numpy_to_pil(np_array):
    return Image.fromarray(np_array)

def pil_to_numpy(pil_img):
    return np.array(pil_img)

def get_feature(fdir):     # get features for folder of images
    mode="legacy_tensorflow"
    model_name="inception_v3"
    num_workers=12
    batch_size=128
    custom_feat_extractor=None
    verbose=True
    custom_image_transform=None
    custom_fn_resize=None
    use_dataparallel=True

    # build the feature extractor based on the mode and the model to be used
    if custom_feat_extractor is None and model_name=="inception_v3":
        feat_model = fid.build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
        # center crop the images
        custom_image_transform = transforms.Compose([
            numpy_to_pil,  # Convert numpy array to PIL
            # transforms.CenterCrop(512),  # Crop operation (works on PIL images)
            pil_to_numpy  # Convert back to numpy
        ])
    elif custom_feat_extractor is None and model_name=="clip_vit_b_32":
        from fid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor

    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir)
    np_feats2 = fid.get_folder_features(fdir, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_transform,
                                    custom_fn_resize=custom_fn_resize)
    return np_feats2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fdir = './MaskGit_src/samples_50000_16'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", type=str, default=fdir, help="Path to image folder 2.")
    parser.add_argument("--log_file", type=str, default='prdc_eval.log', help="Path to log file.")
    
    args = parser.parse_args()
    fdir = args.fdir
    # get number of images in the folder2
    nimages = len(os.listdir(fdir))

    gfile_stream = open(f'{args.log_file}', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    real_file_name = 'VIRTUAL_imagenet256_labeled.npz'
    real_features = np.load(real_file_name)
    # keys_name = list(real_features.keys())
    ### build feature extractor
    mode = "legacy_tensorflow"
    feat_model = fid.build_feature_extractor(mode, device)
    real_features = get_feature_np(real_features['arr_0'], feat_model, mode=mode)

    logger.info(f'Loaded samples from {fdir}')
    fake_features = get_feature(fdir)
    print(real_features.shape, fake_features.shape)

    # https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py#L217
    nearest_k = 3

    metrics = compute_prdc(real_features=real_features,
                        fake_features=fake_features,
                        nearest_k=nearest_k)

    logger.info(f'precision: {metrics["precision"]}, recall: {metrics["recall"]}, density: {metrics["density"]}, coverage: {metrics["coverage"]}')
    logger.info(f'\n\n')

# python prdc.py --fdir ./MaskGit_src/samples_50000_16 --log_file prdc_eval.log
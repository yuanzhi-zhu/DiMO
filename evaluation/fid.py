from cleanfid import fid
import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse
import logging
from omegaconf import OmegaConf
import torch_fidelity
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fdir1 = ''
fdir2 = ''

def numpy_to_pil(np_array):
    return Image.fromarray(np_array)

def pil_to_numpy(pil_img):
    return np.array(pil_img)

def calculate_fid(fdir1, fdir2, ref_stat=None):
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
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor

    if ref_stat is not None:
        print(f"Loading reference statistics from {ref_stat}")
        np_stats = np.load(ref_stat)
        mu1 = np_stats["mu"]
        sigma1 = np_stats["sigma"]
    else:
        print("Calculating reference statistics in the first folder {fdir1}")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = fid.get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                        batch_size=batch_size, device=device, mode=mode,
                                        description=f"FID {fbname1} : ", verbose=verbose,
                                        custom_image_tranform=custom_image_transform,
                                        custom_fn_resize=custom_fn_resize)
        mu1 = np.mean(np_feats1, axis=0)
        sigma1 = np.cov(np_feats1, rowvar=False)
        # save the reference statistics
        np.savez("ref_stat.npz", mu=mu1, sigma=sigma1)

    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = fid.get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_transform,
                                    custom_fn_resize=custom_fn_resize)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    # # save the data statistics
    # np.savez("data_stat.npz", mu2=mu2, sigma2=sigma2)

    score = fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir1", type=str, default=fdir1, help="Path to image folder 1.")
    # Additional arguments for overriding config
    parser.add_argument("--fdir2", type=str, default=fdir2, help="Path to image folder 2.")
    parser.add_argument("--ref_stat", type=str, default=None, help="Path to reference statistics.")
    parser.add_argument("--log_file", type=str, default='fid_eval.log', help="Path to log file.")
    args = parser.parse_args()
    fdir1 = args.fdir1
    fdir2 = args.fdir2
    # get number of images in the folder2
    nimages = len(os.listdir(fdir2))

    gfile_stream = open(f'{args.log_file}', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    metrics_dict = torch_fidelity.calculate_metrics(
            input1=fdir2,
            input2=None,
            fid_statistics_file=args.ref_stat,
            cuda=True,
            isc=True,
            fid=False,
            kid=False,
            prc=False,
            verbose=False,
        )
    # score = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    score = calculate_fid(fdir1, fdir2, ref_stat=args.ref_stat)
    logger.info(f'evaluation on {fdir2} with {nimages} images')
    logger.info(f'FID: {score}, Inception Score: {inception_score}')
    logger.info(f'\n\n')

    # CUDA_VISIBLE_DEVICES=0 python3 evaluation/fid.py \
    #             --fdir2 ./samples/50000 \
    #             --ref_stat ./fid_stats_imagenet256_guided_diffusion.npz \

    # python -m pytorch_fid ./work_dir_4/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr2e-05-amp_fp32-ImageNet/20241022-2156/eval_samples/20241027-2149HR ./FMIR/work_dir_4/sr_bicubic-sf4-guided_unet-t_uniform-sigma0.2-bs32-loss_l1-lr2e-05-amp_fp32-ImageNet/20241022-2156/eval_samples/20241027-2149 --batch-size 1

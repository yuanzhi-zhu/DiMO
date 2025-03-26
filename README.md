# Di[𝙼]O: Distilling Masked Diffusion Models into One-step Generator
<a href='https://yuanzhi-zhu.github.io/DiMO/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2503.15457'><img src='https://img.shields.io/badge/Di[M]O-Arxiv-red'></a>
<a href='https://huggingface.co/Yuanzhi/DiMO'><img src='https://img.shields.io/badge/🤗HuggingFace-Models-orange'></a>
<a href='https://www.alphaxiv.org/overview/2503.15457'><img src='https://img.shields.io/badge/alphaXiv-Blog-blue'></a>

## TLDR
We develop algorithm to distill MDMs into one-step generator, by matching the output distribution of teacher and student model.
<img width="1385" alt="image" src="https://yuanzhi-zhu.github.io/DiMO/static/images/illustration.png" />

## Setup
### Clone and Install
```bash
git clone https://github.com/yuanzhi-zhu/DiMO.git
cd DiMO
pip install -r requirements.txt
```

## Inference Code

#### Download the Distilled Models
```bash
# pwd
# /path/to/DiMO
huggingface-cli download Yuanzhi/DiMO --local-dir models
```

#### Sample Images MaskGit
```bash
network_dir="./models/maskgit"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample_MaskGit.py \
    --save_dir ./samples/test_sample \
    --vit_path $network_dir \
    --mode sample \
    --nb_sample 64
```

#### Sample Images Meissonic
```bash
network_dir='./models/meissonic'
CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        --master_port=29501 \
    sample_Meissonic.py \
    --save_dir ./samples/Meissonic_sample \
    --vit_path $network_dir \
    --mode sample \
```


## Citation

If you find this repo helpful, please cite:

```bibtex
@article{zhu2025di,
      title={Di$\mathtt{[M]}$O: Distilling Masked Diffusion Models into One-step Generator},
      author={Zhu, Yuanzhi and Wang, Xi and Lathuili{\`e}re, St{\'e}phane and Kalogeiton, Vicky},
      journal={arXiv preprint arXiv:2503.15457},
      year={2025}
    }
```


## Acknowledgments
```This work was supported by ANR-22-CE23-0007, ANR-22-CE39-0016, Hi!Paris grant and fellowship, DATAIA Convergence Institute as part of the “Programme d'Investissement d'Avenir” (ANR-17-CONV-0003) operated by Ecole Polytechnique, IP Paris, and was granted access to the IDRIS High-Performance Computing (HPC) resources under the allocation 2024-AD011014300R1 and 2025-AD011015894 made by GENCI and mesoGIP of IP Paris. We also sincerely thank Nacereddine Laddaoui for the help with infrastructure, Haoge Deng and Yao Teng for their insightful discussions that contributed to this work. We are also grateful to Nicolas Dufour, Robin Courant, and Lucas Degeorge for their meticulous proofreading.```

This codebase is based on the [MaskGit PyTorch](https://github.com/valeoai/Halton-MaskGIT/tree/v1.0) and [Meissonic](https://github.com/viiika/Meissonic/).

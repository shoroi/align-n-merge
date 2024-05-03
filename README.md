# Align and Merge Deep Learning Models

This is the official implementation of **CCA Merge** from our paper:

**[Harmony in Diversity: Merging Neural Networks with Canonical Correlation Analysis](https://openreview.net/forum?id=hLuNVjRnY3)**  
Stefan Horoi, Albert Manuel Orozco Camacho, Eugene Belilovsky, Guy Wolf


## Scope of this codebase
Besides providing an official implementation of our CCA Merge method, this repository also aims to partly unify the codebases of past works, thus providing a consistent and fair testing ground for model alignment and merging methods.

The core of this repo was adapted from 
[ZipIt!](https://github.com/gstoica27/ZipIt) [(Stoica et al., ICLR 2024)](https://openreview.net/forum?id=LEYUkvdUhq)
. We have also imported and adapted code from 
[Git Re-Basin](https://github.com/samuela/git-re-basin) [(Ainsworth et al., ICLR 2023)](https://openreview.net/forum?id=CQsmMYmlP5T)
and
[OT Fusion](https://github.com/sidak/otfusion) [(Singh & Jaggi, NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/
to provide a unified PyTorch implementation of these methods allowing fast and easy comparisons. We thank the authors of these works for making their code publicly available.

Our code may differ from that of the original repositories since we have integrated them into a unified framework; for instance, the original Git Re-Basin code uses Jax while our implementation is in PyTorch. Despite this, we have attempted to faithfully replicate the original processes of alignment and merging. Our adaptation does not necessarily include every functionality from the original repositories.


## Using this repo
More details are included in each subdirectory whenever judged necessary.

### Conda environment
Create the conda environment and activate it:
```bash
conda env create --file=requirements.yaml
conda activate align-merge
```

### Training models
The training scripts are located in the ``training`` subdirectory.
```bash
python -m training.train

# ImageNet
python -m training.train_imagenet
# Distributed training
python -m training.train_imagenet --dist-url 'tcp://127.0.0.1:23456' --multiprocessing-distributed --world-size 1 --rank 0
```

### Alignment, merging and evaluation
The scripts for aligning, merging and evaluating models are located in the ``evaluation`` subdirectory.
```bash
# Evaluation of the base models and their ensemble
python -m evaluation.base_ensemle_eval --none-seeds='0_1' --exp-dir="save_dir/cifar10_vgg11x1_logits/none_bs500"
# Evaluation of merged models
python -m evaluation.align_merge_main --merging-fn='match_tensors_identity' --none-seeds='0_1' --exp-dir="save_dir/cifar10_vgg11x1_logits/none_bs500"
```


## Citation
If you use CCA Merge or this codebase in your work, please consider citing:
```
@inproceedings{
horoi2024harmony,
title={Harmony in Diversity: Merging Neural Networks with Canonical Correlation Analysis},
author={Stefan Horoi and Albert Manuel Orozco Camacho and Eugene Belilovsky and Guy Wolf},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=hLuNVjRnY3}
}
```


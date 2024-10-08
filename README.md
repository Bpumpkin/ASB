# ASB: Adaptive search for broad attention based vision transformers
This repository contains the official implementation of Adaptive search for broad attention based vision transformers
![alt](https://github.com/Bpumpkin/ASB/blob/main/figs/evolve2.pdf)


## Model Results
|  Model   | Top1-Acc(%)  |  Params (M)   | Flops(G)  | Model  |
|  ----  | ----  |  ----  | ----  | ----  |
| BViT-5M  | 74.8 |  5.7  | 1.2  | [Download](https://pan.baidu.com/s/1q02tHE9Jk3M9PcIiK4vrdg?pwd=65q1)   |
| BViT-22M  | 81.6 |  22.1  | 4.7  | [Download](https://pan.baidu.com/s/1G_Zh-qDbAtcYvVYLXoF-Nw?pwd=dbgi)   |

## Evaluation
Install PyTorch 1.7.0+ and torchvision 0.8.1+ and timm 0.3.2.

### Training
To evaluate a pre-trained BViT on ImageNet val with a single GPU run:
```
python train_hp.py --eval --resume /path/to/weight --data-path /path/to/imagenet
```

### Training
To train BViT on ImageNet with 4 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_hp.py --arch BViT-5 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```



## Bibtex
If you find BViT is helpful in your project, please consider citing our paper.
```
@article{li2022bvit,
  title={BViT: Broad Attention based Vision Transformer},
  author={Li, Nannan and Chen, Yaran and Li, Weifan and Ding, Zixiang and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2202.06268},
  year={2022}
}
```

## Acknowledgements
The codes are inspired by [timm](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [ViT-torch](https://github.com/lucidrains/vit-pytorch).


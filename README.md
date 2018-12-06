# Train various models on CIFAR10 with Chainer

## Tested environment

- Python 3.6.6 (not tested with Python2)
- pip packages:
    - Chainer v6.0.0b1
    - CuPy v6.0.0b1
    - ChainerCV v0.11.0
    - NumPy v1.15.4
    - matplotlib v3.0.1
    - scikit-image v0.14.0
    - opencv-python 3.4.3
    - tabulate 0.8.2
- GPU related libraries:
    - CUDA v9.2
    - cuDNN: v7.3.1

## Quick Start

```bash
MPLBACKEND=Agg python train.py
```

With full arguments:

```bash
MPLBACKEND=Agg python train.py \
--model_file models/wide_resnet.py \
--model_name WideResNet \
--batchsize 128 \
--training_epoch 500 \
--initial_lr 0.05 \
--lr_decay_rate 0.5 \
--lr_decay_epoch 70 \
--weight_decay 0.0005 \
--random_angle 15.0 \
--pca_sigma 25.5 \
--expand_ratio 1.2 \
--crop_size 28 28 \
--seed 0 \
--gpus 0 
```

## About data augmentation

It performs various data augmentation using [ChainerCV](https://github.com/chainer/chainercv). Provided operations are:

- Random rotating (using OpenCV or scikit-image)
- Random lighting
- Random LR-flipping
- Random zomming (a.k.a. expansion)
- Random cropping

See the details at `transform` function in `train.py`.

## Exprimental Results

| model_name   |   val/main/accuracy |   epoch |   batchsize | crop_size   |   expand_ratio |   pca_sigma |   random_angle |   weight_decay |   initial_lr |   lr_decay_rate |   lr_decay_epoch |
|:-------------|--------------------:|--------:|------------:|:------------|---------------:|------------:|---------------:|---------------:|-------------:|----------------:|-----------------:|
| LeNet5      |            0.860166 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.01 |             0.5 |               50 |
| NIN          |            0.879351 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.01 |             0.5 |              100 |
| VGG          |            0.934237 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.05 |             0.5 |               50 |
| ResNet50     |            0.950455 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.05 |             0.5 |               50 |
| DenseNet     |            0.944818 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.05 |             0.5 |               50 |
| WideResNet   |            0.962322 |     500 |         128 | [28, 28]    |            1.2 |        25.5 |             15 |         0.0005 |         0.05 |             0.5 |               70 |

![](compare.png)

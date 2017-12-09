# Train models on CIFAR10 with Chainer

# Requirement

- Python 2.7.6 + , 3.5.1+
    - Chainer >= 3.1.0
    - ChainerCV >= 0.8.0
    - numpy >= 1.10.1
    - matplotlib >= 2.0.0
    - scikit - image >= 0.13.1
    - opencv-python>=3.3.0
    - tabulate>=0.8.2

# Quick Start

```
$ MPLBACKEND = Agg python train.py
```

# Exprimental Results

|   val/main/accuracy |   epoch | model_name   |   batchsize |   initial_lr |   lr_decay_rate |   lr_decay_epoch |   weight_decay |   random_angle |   pca_sigma |   expand_ratio | crop_size   |
|--------------------:|--------:|:-------------|------------:|-------------:|----------------:|-----------------:|---------------:|---------------:|------------:|---------------:|:------------|
|            0.958169 |     300 | WideResNet   |         128 |         0.05 |             0.5 |               25 |         0.0005 |             15 |        25.5 |            1.2 | [28, 28]    |
|            0.945708 |     300 | ResNet50     |         128 |         0.05 |             0.5 |               25 |         0.0005 |             15 |        25.5 |            1.2 | [28, 28]    |
|            0.930083 |     300 | VGG          |         128 |         0.1  |             0.5 |               30 |         0.0005 |             15 |        25.5 |            1.0 | [28, 28]    |
|            0.9196   |     300 | DenseNet     |         128 |         0.05 |             0.5 |               25 |         0.0005 |             15 |        25.5 |            1.2 | [28, 28]    |
|            0.879351 |     500 | NIN          |         128 |         0.01 |             0.5 |              100 |         0.0005 |             15 |        25.5 |            1.2 | [28, 28]    |
|            0.855815 |     300 | Cifar10      |         128 |         0.01 |             0.5 |               50 |         0.0005 |             15 |        25.5 |            1.2 | [28, 28]    |

![](compare.png)
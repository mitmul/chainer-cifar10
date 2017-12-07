# Train models on CIFAR10 with Chainer

# Requirement

- Python 2.7.6 + , 3.5.1+
    - Chainer >= 3.1.0
    - ChainerCV >= 0.8.0
    - numpy >= 1.10.1
    - matplotlib >= 2.0.0
    - scikit - image >= 0.13.1
    - opencv-python>=3.3.0

# Quick Start

```
$ MPLBACKEND = Agg python train.py
```

# Exprimental Results

| Network | Validation accuracy | Epoch | Batchsize | Random angle | PCA sigma | Expand ratio | Crop size |
|:---|:---|:---|:---|:---|:---|:---|:---|
| ResNet50 | 0.931467592716217 | 100 | 128 | 15.0 | 75.5 | 1.5 | [28, 28] |
| VGG | 0.9300830960273743 | 300 | 128 | 15.0 | 25.5 | 1.0 | [28, 28] |
| WideResNet | 0.9495648741722107 | 109 | 128 | 15.0 | 25.5 | 1.2 | [28, 28] |
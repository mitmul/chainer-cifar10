# Cifar10Net with Chainer

## Download Cifar10 Dataset

```
$ bash download.sh
```

## Start Training

```
$ python train_cpu.py
```

or

```
$ python train_gpu.py
```

### NOTE:

The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) cases divergence of weights.

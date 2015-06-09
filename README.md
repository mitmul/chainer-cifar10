# Cifar10Net with Chainer

## Download Cifar10 Dataset

```
$ bash download.sh
```

## Create Dataset

```
$ python dataset.py
```

## Start Training

```
$ nohup python train.py --gpu 0 --epoch 50 --batchsize 128 --snapshot 10 --datadir data --model vgg --prefix results/vgg_bn_prelu_adam > out.log 2>&1 < /dev/null &
```

You can choose from Cifar10Net(with --model cifar10) or VGGNet(with --model vgg). The architecture of VGGNet is derived from [here](https://github.com/nagadomi/kaggle-cifar10-torch7). Original paper is [here](http://arxiv.org/pdf/1409.1556.pdf).

VGGNet optimized by Adam achieves 87.9% accuracy after 30 epochs.

## Draw Loss Curve

```
$ python draw_loss.py --logfile nohup.out --outfile vgg_loss.jpg
```

### TIPS:

- The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) cases divergence of weights.
- If `model.to_cpu()` and `model.to_gpu()` are called during training, test scores are fixed and never updated despite progress of training loops.

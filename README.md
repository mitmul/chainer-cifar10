# Cifar10Net with Chainer

## Requirement

- Chainer (https://github.com/pfnet/chainer.git)
- progressbar2 (`$ pip install progressbar2`)

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
$ nohup python train.py --gpu 0 --epoch 50 --batchsize 128 --snapshot 10 --datadir data --model vgg --prefix vgg_bn_prelu_adam > out.log 2>&1 < /dev/null &
```

You can choose from Cifar10Net(with --model cifar10) or VGGNet(with --model vgg). The architecture of VGGNet is derived from [here](https://github.com/nagadomi/kaggle-cifar10-torch7). Original paper is [here](http://arxiv.org/pdf/1409.1556.pdf).

## Results

- VGGNet optimized by Adam
    - baseline: 87.9% accuracy after 50 epochs
    - with random horizontal flip and random translation:
        - 90.13% accuracy after 50 epochs
        - [model definition](https://gist.github.com/mitmul/3c7004741e8844f9590a/raw/4418f4853e59ea83633472dba0f9f4497b7af0af/vgg_model.py)
        - [train.py (augmentation settings)](https://gist.github.com/mitmul/3c7004741e8844f9590a/raw/8e0464afa78fabd1724c867b62fb9ebf5fc3b201/train.py)
        - [model file](https://gist.github.com/mitmul/3c7004741e8844f9590a/raw/2ccb56bf35ba235951fb50d653409b7c170cc9ca/vgg_aug_epoch_50.chainermodel)

## Draw Loss Curve

```
$ python draw_loss.py --logfile nohup.out --outfile vgg_loss.jpg
```

![loss curve](loss.jpg)

### TIPS:

- The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) cases divergence of weights.
- If `model.to_cpu()` and `model.to_gpu()` are called during training, test scores are fixed and never updated despite progress of training loops.

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


## Results

- VGGNet optimized by Adam
    - baseline: 87.9% accuracy after 50 epochs
    - with random horizontal flip: 90.0% accuracy after 50 epochs
        - [model definition](https://gist.githubusercontent.com/mitmul/87fcc1601d59f6fa928f/raw/1a293f6c5a846a6165f38b5c1ddc49b9ec47595a/vgg_mode.py)
        - [model file](https://gist.github.com/mitmul/87fcc1601d59f6fa928f/raw/093a0daff924740a28f71b4c1c580d34b1de1bf6/vgg_epoch_50.chainermodel)

## Draw Loss Curve

```
$ python draw_loss.py --logfile nohup.out --outfile vgg_loss.jpg
```

![loss curve](http://bit.ly/1e2LLWT)

### TIPS:

- The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) cases divergence of weights.
- If `model.to_cpu()` and `model.to_gpu()` are called during training, test scores are fixed and never updated despite progress of training loops.

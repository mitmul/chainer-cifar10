# Cifar10Net with Chainer

## Download Cifar10 Dataset

```
$ bash download.sh
```

## Create Dataset

- with whitening
    `$ python dataset.py --whitening True`
- without whitening
    `$ python dataset.py --whitening False`

## Start Training

```
$ nohup python train.py --model vgg --gpu 0 --epoch 50 --batchsize 128 --prefix vgg &
```

You can choose from Cifar10Net(with --model cifar10) or VGGNet(with --model vgg).

## Draw Loss Curve

```
$ python draw_loss.py --logfile nohup.out --outfile vgg_loss.jpg
```

### NOTE:

The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) cases divergence of weights.

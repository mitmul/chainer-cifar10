# chainer-cifar10

## Requirement
- Chainer v1.5.0.2
- scikit-image 0.11.3
- scipy 0.16.0
- numpy 1.10. 1

## Download & Construct Cifar10 Dataset

```
$ bash download.sh
```

## Start Training

```
$ nohup python train.py &
```

## Draw Loss Curve

```
$ python draw_loss.py --logfile log.txt --outfile log.jpg
```

## Deep Residual Network (ResNet-110)

```
$ python dataset.py --whitening 0
$ python train.py --model models/ResNet.py --lr 0.1 --gpu 0
```

The test accuracy after 15 epochs is 0.9406 (error (%): 5.94). The test accuracy reported in the MSR paper ([Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)) is 0.9357 (error (%): 6.43) (see Table 6).

### Resulting loss curve

![](https://raw.githubusercontent.com/wiki/mitmul/chainer-cifar10/images/ResNet_loss.png)

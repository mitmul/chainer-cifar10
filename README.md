# chainer-cifar10
## Requirement
- [Chainer v1.5](http://chainer.org)
- scikit-image 0.11.3
- scipy 0.16.0
- numpy 1.10. 1

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
$ python train.py
```

## Test

```
$ python test.py --eval normal \
--model results/VGG_mini_ABN/VGG_mini_ABN.py \
--param results/VGG_mini_ABN/VGG_mini_ABN_Adam_epoch_100.chainermodel \
--norm 0 --batchsize 128 --gpu 0
```

## Draw Loss Curve

```
$ python draw_loss.py --logfile log.txt --outfile log.jpg
```

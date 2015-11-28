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

# CIFAR-10 with Chainer

## Requirement

- [Chainer v1.5](http://chainer.org)
- progressbar2
    - `$ pip install progressbar2`

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

### TIPS:

- The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) causes divergence of weights.
- If `model.to_cpu()` and `model.to_gpu()` are called during training, test scores are fixed and never updated.

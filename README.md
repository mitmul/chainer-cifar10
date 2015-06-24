# Cifar10Net with Chainer

## Requirement

- [Chainer](http://chainer.org)
    - `$ git clone https://github.com/pfnet/chainer.git`
    - `$ cd chainer; python setup.py install`
    - `$ pip install chainer-cuda-deps`
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

See the help messages with --help option for details.

You can choose model file to be trained from models dir. Cifar10, Network In Network (NIN), and VGG-net (with some variants) are already prepared. The architecture of VGG_mini is derived from [here](https://github.com/nagadomi/kaggle-cifar10-torch7). The original paper of VGG-net is found in [here](http://arxiv.org/pdf/1409.1556.pdf).
<!-- 
## Results

- VGG_mini model train by MomentumSGD (with all default values) shows the below results
    - -->

![loss curve](loss.jpg)

## Draw Loss Curve

```
$ python draw_loss.py --logfile log.txt --outfile log.jpg
```

### TIPS:

- The label vector will be passed to softmax_cross_entropy should be in shape (N,). (N, 1) causes divergence of weights.
- If `model.to_cpu()` and `model.to_gpu()` are called during training, test scores are fixed and never updated.

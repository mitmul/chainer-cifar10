# Cifar10Net with Chainer

## Download Cifar10 Dataset

```
$ bash download.sh
```

## Start Training

```
$ python train_cpu.py
```

### NOTE:

Now the weight values of conv1 layer diverge.

#### LOG:

```
iteration      0:	accuracy=0.109	conv1_maxval=0.416:
iteration    128:	accuracy=0.116	conv1_maxval=0.416:
iteration    256:	accuracy=0.097	conv1_maxval=0.416:
iteration    384:	accuracy=0.088	conv1_maxval=0.416:
iteration    512:	accuracy=0.112	conv1_maxval=0.416:
iteration    640:	accuracy=0.098	conv1_maxval=0.416:
iteration    768:	accuracy=0.071	conv1_maxval=0.413:
iteration    896:	accuracy=0.086	conv1_maxval=0.452:
/home/ubuntu/anaconda/lib/python2.7/site-packages/chainer-1.0.0-py2.7.egg/chainer/functions/softmax_cross_entropy.py:14: RuntimeWarning: divide by zero encountered in log
  return -numpy.log(self.y[xrange(len(t)), t]).sum(keepdims=True) / t.size,
iteration   1024:	accuracy=0.101	conv1_maxval=0.757:
iteration   1152:	accuracy=0.109	conv1_maxval=1.226:
iteration   1280:	accuracy=0.102	conv1_maxval=1.837:
iteration   1408:	accuracy=0.102	conv1_maxval=13.982:
iteration   1536:	accuracy=0.086	conv1_maxval=590.896:
iteration   1664:	accuracy=0.062	conv1_maxval=3899858.750:
iteration   1792:	accuracy=0.069	conv1_maxval=687467397120.000:
/home/ubuntu/anaconda/lib/python2.7/site-packages/numpy/core/_methods.py:106: RuntimeWarning: overflow encountered in multiply
  x = um.multiply(x, x, out=x)
iteration   1920:	accuracy=0.117	conv1_maxval=1306190938112.000:
/home/ubuntu/anaconda/lib/python2.7/site-packages/chainer-1.0.0-py2.7.egg/chainer/functions/softmax.py:17: RuntimeWarning: invalid value encountered in subtract
  self.y = x[0] - numpy.amax(x[0], axis=1, keepdims=True)
iteration   2048:	accuracy=0.078	conv1_maxval=nan:
/home/ubuntu/anaconda/lib/python2.7/site-packages/chainer-1.0.0-py2.7.egg/chainer/functions/relu.py:33: RuntimeWarning: invalid value encountered in greater
  return gy[0] * (x[0] > 0),
iteration   2176:	accuracy=0.070	conv1_maxval=nan:
iteration   2304:	accuracy=0.070	conv1_maxval=nan:
iteration   2432:	accuracy=0.117	conv1_maxval=nan:
iteration   2560:	accuracy=0.086	conv1_maxval=nan:
iteration   2688:	accuracy=0.078	conv1_maxval=nan:
iteration   2816:	accuracy=0.094	conv1_maxval=nan:
iteration   2944:	accuracy=0.117	conv1_maxval=nan:
iteration   3072:	accuracy=0.070	conv1_maxval=nan:
iteration   3200:	accuracy=0.180	conv1_maxval=nan:
iteration   3328:	accuracy=0.133	conv1_maxval=nan:
iteration   3456:	accuracy=0.125	conv1_maxval=nan:
iteration   3584:	accuracy=0.094	conv1_maxval=nan:
iteration   3712:	accuracy=0.180	conv1_maxval=nan:
iteration   3840:	accuracy=0.125	conv1_maxval=nan:
iteration   3968:	accuracy=0.086	conv1_maxval=nan:
iteration   4096:	accuracy=0.078	conv1_maxval=nan:
iteration   4224:	accuracy=0.055	conv1_maxval=nan:
iteration   4352:	accuracy=0.109	conv1_maxval=nan:
iteration   4480:	accuracy=0.117	conv1_maxval=nan:
iteration   4608:	accuracy=0.102	conv1_maxval=nan:
iteration   4736:	accuracy=0.102	conv1_maxval=nan:
iteration   4864:	accuracy=0.109	conv1_maxval=nan:
iteration   4992:	accuracy=0.125	conv1_maxval=nan:
iteration   5120:	accuracy=0.117	conv1_maxval=nan:
iteration   5248:	accuracy=0.102	conv1_maxval=nan:
iteration   5376:	accuracy=0.133	conv1_maxval=nan:
iteration   5504:	accuracy=0.109	conv1_maxval=nan:
iteration   5632:	accuracy=0.062	conv1_maxval=nan:
iteration   5760:	accuracy=0.117	conv1_maxval=nan:
```

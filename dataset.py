#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

from chainer.datasets import cifar
from chainer.datasets import TransformDataset
import numpy as np

import matplotlib.pyplot as plt
import train as train_module

train_orig, _ = cifar.get_cifar10(scale=255.)
mean = np.mean([x for x, _ in train_orig], axis=(0, 2, 3))
std = np.std([x for x, _ in train_orig], axis=(0, 2, 3))
print('mean:', mean)
print('std:', std)

train_transform = partial(train_module.transform,
                          mean=mean, std=std, train=True)
train = TransformDataset(train_orig, train_transform)

for i, (img, label) in enumerate(train[:10]):
    img_orig, _ = train_orig[i]
    print(img_orig.min(), img_orig.max())
    img *= std[:, None, None]
    img += mean[:, None, None]
    img = np.clip(img / 255., 0, 1)

    plt.imshow(img.transpose(1, 2, 0))
    plt.savefig('img{}.png'.format(i))
    plt.close()

    plt.imshow(img_orig.transpose(1, 2, 0) / 255.)
    plt.savefig('img{}_orig.png'.format(i))
    plt.close()

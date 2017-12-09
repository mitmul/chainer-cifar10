#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import os

import numpy as np

from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import matplotlib.pyplot as plt
import train as train_module

train_orig, _ = cifar.get_cifar10(scale=255.)
mean = np.mean([x for x, _ in train_orig], axis=(0, 2, 3))
std = np.std([x for x, _ in train_orig], axis=(0, 2, 3))
print('mean:', mean)
print('std:', std)

random_angle = 45.0
pca_sigma = 25.5
expand_ratio = 1.5
crop_size = [28, 28]

train_transform = partial(
    train_module.transform, mean=mean, std=std, random_angle=random_angle,
    pca_sigma=pca_sigma, expand_ratio=expand_ratio, crop_size=crop_size,
    train=True)
train = TransformDataset(train_orig, train_transform)

out_dir = 'test_images'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for i, (img, label) in enumerate(train[:100]):
    img_orig, _ = train_orig[i]
    img *= std[:, None, None]
    img += mean[:, None, None]
    img = np.clip(img / 255., 0, 1)

    plt.imshow(img.transpose(1, 2, 0))
    plt.savefig('{}/img{}.png'.format(out_dir, i))
    plt.close()

    plt.imshow(img_orig.transpose(1, 2, 0) / 255.)
    plt.savefig('{}/img{}_orig.png'.format(out_dir, i))
    plt.close()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from functools import partial
from importlib import import_module
import os
import random
import re
import time

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
from chainer.training import extensions
import numpy as np
import six

from chainercv import transforms


def transform(inputs, train=True):
    img, label = inputs

    # Color augmentation
    if train:
        img = transforms.pca_lighting(img, 25.5)
        img = transforms.random_flip(img)

    # Per-image standardization
    img -= img.mean(axis=(1, 2))[:, None, None]
    img /= img.std(axis=(1, 2))[:, None, None] + 1e-8

    # Random expand
    if train:
        img = transforms.random_expand(img, max_ratio=1.5)
        img = transforms.random_crop(img, (32, 32))

    return img, label


def create_result_dir(prefix):
    result_dir = 'results/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def run_training(net, train, valid, result_dir, batchsize=64, devices=-1):
    # Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(valid, batchsize, False, False)

    # Model
    net = L.Classifier(net)

    # Optimizer
    optimizer = optimizers.MomentumSGD(lr=0.1)
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # Updater
    if isinstance(devices, int):
        devices['main'] = devices
        updater = training.StandardUpdater(
            train_iter, optimizer, device=devices)
    elif isinstance(devices, dict):
        updater = training.ParallelUpdater(
            train_iter, optimizer, devices=devices)

    # 6. Trainer
    trainer = training.Trainer(updater, (100, 'epoch'), out=result_dir)

    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(
        test_iter, net, device=devices['main']), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
        file_name='accuracy.png'))
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1), trigger=(30, 'epoch'))
    trainer.run()

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='models/ResNet.py')
    parser.add_argument('--model_name', type=str, default='ResNet')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0])
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if len(args.gpus) > 1 or args.gpus[0] >= 0:
        chainer.cuda.cupy.seed(args.seed)

    # Load model
    ext = os.path.splitext(args.model_file)[1]
    mod_path = '.'.join(os.path.split(args.model_file)).replace(ext, '')
    mod = import_module(mod_path)
    model = getattr(mod, args.model_name)(10)

    # create result dir
    result_dir = create_result_dir(
        os.path.splitext(os.path.basename(args.model))[0])

    train, valid = cifar.get_cifar10(scale=255.)

    train = TransformDataset(train, partial(transform, train=True))
    valid = TransformDataset(valid, partial(transform, train=False))

    run_training(net, train, valid, result_dir, args.batchsize, args.gpus)

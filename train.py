#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from functools import partial
from importlib import import_module
import json
import os
import random
import re
import shutil
import time

import chainer
from chainer.datasets import cifar
from chainer.datasets import TransformDataset
from chainer import iterators
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import numpy as np

from chainercv import transforms


def transform(
        inputs, mean, std, pca_sigma, expand_ratio, crop_size, train=True):
    img, label = inputs
    img = img.copy()

    # Color augmentation and Flipping
    if train:
        img = transforms.pca_lighting(img, pca_sigma)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    # Random crop
    if train:
        img = transforms.random_flip(img, x_random=True)
        img = transforms.random_expand(img, max_ratio=expand_ratio)
        img = transforms.random_crop(img, tuple(crop_size))

    return img, label


def create_result_dir(prefix):
    result_dir = 'results/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
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
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)

    # Data augmentation settings
    parser.add_argument('--pca_sigma', type=float, default=51)
    parser.add_argument('--expand_ratio', type=float, default=1.5)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])
    args = parser.parse_args()

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    if len(args.gpus) > 1 or args.gpus[0] >= 0:
        chainer.cuda.cupy.random.seed(args.seed)
        gpus = {'main': args.gpus[0]}
        if len(args.gpus) > 1:
            gpus.update({'gpu{}'.format(i): i for i in args.gpus[1:]})
        args.gpus = gpus

    # Load model
    ext = os.path.splitext(args.model_file)[1]
    mod_path = '.'.join(os.path.split(args.model_file)).replace(ext, '')
    mod = import_module(mod_path)
    net = getattr(mod, args.model_name)(10)

    # create result dir
    result_dir = create_result_dir(args.model_name)
    shutil.copy(args.model_file, os.path.join(
        result_dir, os.path.basename(args.model_file)))
    with open(os.path.join(result_dir, 'args'), 'w') as fp:
        fp.write(json.dumps(vars(args)))

    train, valid = cifar.get_cifar10(scale=255.)
    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))

    train_transform = partial(
        transform, mean=mean, std=std, pca_sigma=args.pca_sigma,
        expand_ratio=args.expand_ratio, crop_size=args.crop_size, train=True)
    valid_transform = partial(transform, mean=mean, std=std, train=False)

    train = TransformDataset(train, train_transform)
    valid = TransformDataset(valid, valid_transform)

    run_training(net, train, valid, result_dir, args.batchsize, args.gpus)

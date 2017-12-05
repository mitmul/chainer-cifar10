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
import cv2 as cv


# scikit-image's rotate function is almost 7x slower than OpenCV
def cv_rotate(img, angle):
    center = (img.shape[0] // 2, img.shape[1] // 2)
    r = cv.getRotationMatrix2D(center, angle, 1.0)
    img = cv.warpAffine(img, r, img.shape[:2])
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0,
        crop_size=(32, 32), train=True):
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0.0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = img.transpose(1, 2, 0) / 255.
        img = cv_rotate(img, angle)
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)

    # Color augmentation and Flipping
    if train and pca_sigma != 255.:
        img = transforms.pca_lighting(img, pca_sigma)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1.0:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
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
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--pca_sigma', type=float, default=75.5)
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
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    train, valid = cifar.get_cifar10(scale=255.)
    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))

    train_transform = partial(
        transform, mean=mean, std=std, random_angle=args.random_angle,
        pca_sigma=args.pca_sigma, expand_ratio=args.expand_ratio,
        crop_size=args.crop_size, train=True)
    valid_transform = partial(transform, mean=mean, std=std, train=False)

    train = TransformDataset(train, train_transform)
    valid = TransformDataset(valid, valid_transform)

    run_training(net, train, valid, result_dir, args.batchsize, args.gpus)

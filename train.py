#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from chainer import optimizers, cuda
from dataset import load_dataset
import cPickle as pickle
import argparse
import os
import sys
sys.path.append('models')

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='cifar10',
                    choices=['cifar10', 'vgg'])
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument('--epoch', '-e', type=int, default=20)
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--datadir', '-d', type=str, default='data')
parser.add_argument('--prefix', '-p', type=str)
args = parser.parse_args()
print(args)

snapshot_dir, prefix = os.path.split(args.prefix)
if snapshot_dir and not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)

if args.model == 'cifar10':
    from cifar10_model import Cifar10Net as Net
if args.model == 'vgg':
    from vgg_model import VGGNet as Net

# load dataset
train_data, train_labels, test_data, test_labels = load_dataset(args.datadir)
N = train_data.shape[0]
N_test = test_data.shape[0]

# prepare model and optimizer
model = Net()
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# learning loop
n_epoch = args.epoch
batchsize = args.batchsize
for epoch in range(1, n_epoch + 1):

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = train_data[perm[i:i + batchsize]]
        y_batch = train_labels[perm[i:i + batchsize]]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        if args.gpu < 0:
            print('train loss={}'.format(loss.data * batchsize))

    print('epoch:{}\ttrain mean loss={}, accuracy={}'.format(
        epoch, sum_loss / N, sum_accuracy / N))

    # evaluation
    if epoch % 5 == 0:
        sum_accuracy = 0
        sum_loss = 0
        for i in xrange(0, N_test, batchsize):
            x_batch = test_data[i:i + batchsize]
            y_batch = test_labels[i:i + batchsize]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            loss, acc = model.forward(x_batch, y_batch, train=False)
            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print('epoch:{}\ttest mean loss={}, accuracy={}'.format(
            epoch, sum_loss / N_test, sum_accuracy / N_test))

        model_fn = '%s_epoch_%d.chainermodel' % (args.prefix, epoch)
        pickle.dump(model, open(model_fn, 'wb'), -1)

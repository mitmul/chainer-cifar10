#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import optimizers, cuda
from dataset import load_dataset
import cPickle as pickle
import argparse
import time
import os
import sys
import imp

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, default='cifar10_model')
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument('--epoch', '-e', type=int, default=20)
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--prefix', '-p', type=str)
parser.add_argument('--snapshot', '-s', type=int, default=5)
parser.add_argument('--restart_from', '-r', type=str)
parser.add_argument('--epoch_offset', '-o', type=int, default=0)
parser.add_argument('--datadir', '-d', type=str, default='data')
args = parser.parse_args()
print(args)

snapshot_dir, prefix = os.path.split(args.prefix)
if snapshot_dir and not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)

model_n = ''.join([n[0].upper() + n[1:] for n in args.model.split('_')])
module = imp.load_source(model_n, 'models/%s_model.py' % args.model)
Net = getattr(module, model_n)

# load dataset
train_data, train_labels, test_data, test_labels = load_dataset(args.datadir)
N = train_data.shape[0]
N_test = test_data.shape[0]

# prepare model and optimizer
model = Net()
if args.restart_from is not None:
    if args.gpu >= 0:
        cuda.init(args.gpu)
    model = pickle.load(open(args.restart_from, 'rb'))
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

optimizer = optimizers.Adam()
# optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
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
        # data augmentation
        for j, x in enumerate(x_batch):
            # LR-flip
            if np.random.randint(2) == 1:
                x = np.fliplr(x.transpose((1, 2, 0)))
                x_batch[j] = x.transpose((2, 0, 1))
            # # Global Contrast Normalization
            # for ch, x in enumerate(x_batch[j]):
            #     x_batch[j][ch] = (x - np.mean(x)) / np.std(x)
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch)
        loss.backward()
        # optimizer.weight_decay(0.0001)
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        if args.gpu < 0:
            print('train loss={}'.format(loss.data * batchsize))

    print('lr:{}'.format(optimizer.lr))
    print('epoch:{:02d}\ttrain mean loss={}, accuracy={}'.format(
        epoch + args.epoch_offset, sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = test_data[i:i + batchsize]
        y_batch = test_labels[i:i + batchsize]
        # for j, x in enumerate(x_batch):
        #     # Global Contrast Normalization
        #     for ch, x in enumerate(x_batch[j]):
        #         x_batch[j][ch] = (x - np.mean(x)) / np.std(x)

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = model.forward(x_batch, y_batch, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('epoch:{:02d}\ttest mean loss={}, accuracy={}'.format(
        epoch + args.epoch_offset, sum_loss / N_test, sum_accuracy / N_test))

    if epoch == 1 or epoch % args.snapshot == 0:
        model_fn = '%s_epoch_%d.chainermodel' % (
            args.prefix, epoch + args.epoch_offset)
        pickle.dump(model, open(model_fn, 'wb'), -1)

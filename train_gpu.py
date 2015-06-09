#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
from chainer import optimizers, cuda
from dataset import load_dataset
from cifar10_model import Cifar10Net

if os.path.exists('loss_gpu.txt'):
    os.remove('loss_gpu.txt')

cuda.init(0)

# load dataset
train_data, train_labels, test_data, test_labels = load_dataset()
N = train_data.shape[0]
N_test = test_data.shape[0]

# prepare model and optimizer
model = Cifar10Net()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(model.collect_parameters())

# learning loop
n_epoch = 100
batchsize = 128
for epoch in range(1, n_epoch + 1):

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = cuda.to_gpu(train_data[perm[i:i + batchsize]])
        y_batch = cuda.to_gpu(train_labels[perm[i:i + batchsize]])
        # print('train_data:{}'.format(train_data[i:i + batchsize].shape))
        # print('train_labels:{}'.format(train_labels[i:i + batchsize].shape))

        optimizer.zero_grads()
        loss, acc, pred = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.weight_decay(0.004)
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    fp = open('loss_gpu.txt', 'a')
    print('epoch:{}\ttrain mean loss={}, accuracy={}'.format(
        epoch, sum_loss / N, sum_accuracy / N), file=fp)

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = cuda.to_gpu(test_data[i:i + batchsize])
        y_batch = cuda.to_gpu(test_labels[i:i + batchsize])
        # print('test_data:{}'.format(test_data[i:i + batchsize].shape))
        # print('test_labels:{}'.format(test_labels[i:i + batchsize].shape))

        loss, acc, pred = model.forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print('epoch:{}\ttest mean loss={}, accuracy={}'.format(
        epoch, sum_loss / N_test, sum_accuracy / N_test), file=fp)

    print(epoch)

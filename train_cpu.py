#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chainer import optimizers
from dataset import load_dataset
from cifar10_model import Cifar10Net

# load dataset
train_data, train_labels, test_data, test_labels = load_dataset()
N = train_data.shape[0]

model = Cifar10Net()

# prepare model and optimizer
optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(model.collect_parameters())

# training
batchsize = 128
sum_accuracy = 0
sum_loss = 0

perm = np.random.permutation(N)
for i in range(0, N, batchsize):
    x_batch = train_data[perm[i:i + batchsize]]
    y_batch = train_labels[perm[i:i + batchsize]]

    loss, acc, pred = model.forward(x_batch, y_batch)

    optimizer.zero_grads()
    loss.backward()
    optimizer.weight_decay(0.004)
    optimizer.update()

    sum_loss += float(loss.data) * batchsize
    sum_accuracy += float(acc.data) * batchsize

    print('iteration {:6d}:\taccuracy={:.5f}\tconv1_maxval={:.5f}:'
          .format(i, acc.data, model.conv1.W.max()))

print('train mean loss={}\taccuracy={}'
      .format(sum_loss / N, sum_accuracy / N))

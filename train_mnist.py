#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 100
n_epoch = 20
n_units = 1000

# Prepare dataset
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist.data,   [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size
x_train = x_train.reshape((N, 1, 28, 28))
x_test = x_test.reshape((N_test, 1, 28, 28))

x_train = np.repeat(x_train, 3, axis=1)
x_test = np.repeat(x_test, 3, axis=1)

print x_train.shape, x_train.dtype
print x_test.shape, x_test.dtype
print y_train.shape, y_train.dtype

# Prepare multi-layer perceptron model
model = FunctionSet(
    conv1=F.Convolution2D(3, 32, 5, stride=1, pad=2),
    bn1=F.BatchNormalization(32),
    conv2=F.Convolution2D(32, 32, 5, stride=1, pad=2),
    bn2=F.BatchNormalization(32),
    conv3=F.Convolution2D(32, 64, 5, stride=1, pad=2),
    fc4=F.Linear(576, 10)
)

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# Neural net architecture


def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h = F.max_pooling_2d(F.relu(model.bn1(model.conv1(x))), 3, stride=2)
    h = F.max_pooling_2d(F.relu(model.bn2(model.conv2(h))), 3, stride=2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 3, stride=2)
    h = model.fc4(h)

    return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# Learning loop
for epoch in xrange(1, n_epoch + 1):
    print 'epoch', epoch

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i + batchsize]]
        y_batch = y_train[perm[i:i + batchsize]]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, acc, pred = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print('iteration {:6d}:\taccuracy={:.5f}\tconv1_maxval={:.5f}:'
              .format(i, acc.data, model.conv1.W.max()))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print 'test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import math


class Module(chainer.Chain):

    def __init__(self, n_in, n_out, stride=1):
        w = math.sqrt(2)
        super(Module, self).__init__(
            conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, w, nobias=True),
            bn1=L.BatchNormalization(n_out),
            conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(n_out),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
            if x.data.shape[1] != h.data.shape[1]:
                x = F.concat((x, x * 0))
        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, n_in, n_out, n, stride=1):
        super(Block, self).__init__()
        links = [('m0', Module(n_in, n_out, stride))]
        links += [('m{}'.format(i + 1), Module(n_out, n_out))
                  for i in range(n - 1)]
        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            x = getattr(self, name)(x, train)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n=18):
        super(ResNet, self).__init__()
        w = math.sqrt(2)
        links = [('conv1', L.Convolution2D(3, 16, 3, 1, 0, w)),
                 ('bn2', L.BatchNormalization(16)),
                 ('_relu3', F.ReLU()),
                 ('res4', Block(16, 32, n)),
                 ('res5', Block(32, 64, n, 2)),
                 ('res6', Block(64, 64, n, 2)),
                 ('_apool7', F.AveragePooling2D(6, 1, 0, False, True)),
                 ('fc8', L.Linear(64, 10))]
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        for name, f in self.forward:
            if 'res' in name:
                x = getattr(self, name)(x, self.train)
            elif name.startswith('bn'):
                x = getattr(self, name)(x, not self.train)
            elif name.startswith('_'):
                x = f(x)
            else:
                x = getattr(self, name)(x)
        if self.train:
            self.loss = F.softmax_cross_entropy(x, t)
            self.accuracy = F.accuracy(x, t)
            return self.loss
        else:
            return x

model = ResNet()

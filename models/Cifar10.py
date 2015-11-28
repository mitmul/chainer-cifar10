#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class Cifar10(chainer.Chain):

    def __init__(self):
        super(Cifar10, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=1, pad=2),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            bn2=L.BatchNormalization(32),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(1024, 10)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.bn1(self.conv1(x), test=not self.train))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.bn2(self.conv2(h), test=not self.train))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.fc4(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        if self.train:
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

model = Cifar10()

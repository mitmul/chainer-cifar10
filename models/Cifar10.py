#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class Cifar10(chainer.Chain):

    def __init__(self):
        super(Cifar10, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(1344, 4096),
            fc5=F.Linear(4096, 10),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5, train=self.train)
        h = self.fc5(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        if self.train:
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

model = Cifar10()

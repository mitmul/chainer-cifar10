#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class Cifar10_SPP(FunctionSet):

    def __init__(self):
        super(Cifar10_SPP, self).__init__(
            conv1=F.Convolution2D(3, 32, 5, stride=1, pad=2),
            bn1=F.BatchNormalization(32),
            conv2=F.Convolution2D(32, 32, 5, stride=1, pad=2),
            bn2=F.BatchNormalization(32),
            conv3=F.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=F.Linear(1344, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = self.fc4(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

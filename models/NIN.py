#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import math


class NIN(chainer.Chain):

    def __init__(self):
        w = math.sqrt(2)
        super(NIN, self).__init__(
            mlpconv1=L.MLPConvolution2D(
                3, (192, 160, 96), 5, pad=2, wscale=w),
            mlpconv2=L.MLPConvolution2D(
                96, (192, 192, 192), 5, pad=2, wscale=w),
            mlpconv3=L.MLPConvolution2D(
                192, (192, 192, 10), 3, pad=1, wscale=w),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.mlpconv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = F.relu(self.mlpconv2(h))
        h = F.average_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = self.mlpconv3(h)
        h = F.average_pooling_2d(h, h.data.shape[2])
        self.y = F.reshape(h, (x.data.shape[0], 10))
        self.pred = F.softmax(self.y)

        self.loss = F.softmax_cross_entropy(self.y, t)
        self.accuracy = F.accuracy(self.y, t)

        if self.train:
            return self.loss
        else:
            return self.pred

model = NIN()

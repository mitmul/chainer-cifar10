#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import chainer
import chainer.functions as F
import chainer.links as L


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            mlpconv1=L.MLPConvolution2D(
                3, (96, 96, 96), 3, stride=1, pad=1, wscale=w),
            mlpconv2=L.MLPConvolution2D(
                96, (256, 256, 256), 3, pad=1, wscale=w),
            mlpconv3=L.MLPConvolution2D(
                256, (384, 384, 384), 3, pad=1, wscale=w),
            mlpconv4=L.MLPConvolution2D(
                384, (1024, 1024, 10), 3, pad=1, wscale=w),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 2, stride=2)
        print(h.data.shape)
        h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 2, stride=2)
        print(h.data.shape)
        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 2, stride=2)
        print(h.data.shape)
        h = self.mlpconv4(F.dropout(h, train=self.train))
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 10))

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        return self.loss

model = NIN()

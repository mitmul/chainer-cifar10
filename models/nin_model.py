#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class Nin(FunctionSet):

    """
    Network In Network
    """

    def __init__(self):
        super(Nin, self).__init__(
            conv1=F.Convolution2D(3, 192, 5, stride=1, pad=2),
            bn1=F.BatchNormalization(192, decay=0.9, eps=1e-5),
            prelu1=F.PReLU(),

            conv2=F.Convolution2D(192, 160, 1, stride=1, pad=0),
            bn2=F.BatchNormalization(160, decay=0.9, eps=1e-5),
            prelu2=F.PReLU(),

            conv3=F.Convolution2D(160, 96, 1, stride=1, pad=0),
            bn3=F.BatchNormalization(96, decay=0.9, eps=1e-5),
            prelu3=F.PReLU(),

            conv4=F.Convolution2D(96, 192, 5, stride=1, pad=2),
            bn4=F.BatchNormalization(192, decay=0.9, eps=1e-5),
            prelu4=F.PReLU(),

            conv5=F.Convolution2D(192, 192, 1, stride=1, pad=0),
            bn5=F.BatchNormalization(192, decay=0.9, eps=1e-5),
            prelu5=F.PReLU(),

            conv6=F.Convolution2D(192, 192, 1, stride=1, pad=0),
            bn6=F.BatchNormalization(192, decay=0.9, eps=1e-5),
            prelu6=F.PReLU(),

            conv7=F.Convolution2D(192, 192, 1, stride=1, pad=0),
            bn7=F.BatchNormalization(192, decay=0.9, eps=1e-5),
            prelu7=F.PReLU(),

            conv8=F.Convolution2D(192, 10, 1, stride=1, pad=0),
            prelu8=F.PReLU(),
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)

        h = self.prelu1(self.bn1(self.conv1(x)))
        h = self.prelu2(self.bn2(self.conv2(h)))
        h = self.prelu3(self.bn3(self.conv3(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5, train=train)

        h = self.prelu4(self.bn4(self.conv4(h)))
        h = self.prelu5(self.bn5(self.conv5(h)))
        h = self.prelu6(self.bn6(self.conv6(h)))
        h = F.average_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5, train=train)

        h = self.prelu7(self.bn7(self.conv7(h)))
        h = self.prelu8(self.conv8(h))
        h = F.average_pooling_2d(h, 7, stride=1)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

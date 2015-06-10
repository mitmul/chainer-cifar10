#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class MiniVgg(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(MiniVgg, self).__init__(
            conv1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1=F.BatchNormalization(64),
            prelu1=F.PReLU(),

            conv2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn2=F.BatchNormalization(64),
            prelu2=F.PReLU(),

            conv3=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn3=F.BatchNormalization(128),
            prelu3=F.PReLU(),

            conv4=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn4=F.BatchNormalization(128),
            prelu4=F.PReLU(),

            conv5=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn5=F.BatchNormalization(128),
            prelu5=F.PReLU(),

            conv6=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn6=F.BatchNormalization(128),
            prelu6=F.PReLU(),

            fc7=F.Linear(2048, 1024),
            prelu7=F.PReLU(),

            fc8=F.Linear(1024, 1024),
            prelu8=F.PReLU(),

            fc9=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)

        h = self.prelu1(self.bn1(self.conv1(x)))
        h = self.prelu2(self.bn2(self.conv2(h)))
        h = F.dropout(
            F.max_pooling_2d(h, 3, stride=2), train=train, ratio=0.25)

        h = self.prelu3(self.bn3(self.conv3(h)))
        h = self.prelu4(self.bn4(self.conv4(h)))
        h = F.dropout(
            F.max_pooling_2d(h, 3, stride=2), train=train, ratio=0.25)

        h = self.prelu5(self.bn5(self.conv5(h)))
        h = self.prelu6(self.bn6(self.conv6(h)))
        h = F.dropout(
            F.max_pooling_2d(h, 3, stride=2), train=train, ratio=0.25)

        h = F.dropout(self.prelu7(self.fc7(h)), train=train, ratio=0.5)
        h = F.dropout(self.prelu8(self.fc8(h)), train=train, ratio=0.5)
        h = self.fc9(h)

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class Vgg(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(Vgg, self).__init__(
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

            conv5=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn5=F.BatchNormalization(256),
            prelu5=F.PReLU(),

            conv6=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn6=F.BatchNormalization(256),
            prelu6=F.PReLU(),

            conv7=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn7=F.BatchNormalization(256),
            prelu7=F.PReLU(),

            conv8=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn8=F.BatchNormalization(256),
            prelu8=F.PReLU(),

            fc9=F.Linear(4096, 1024),
            prelu9=F.PReLU(),

            fc10=F.Linear(1024, 1024),
            prelu10=F.PReLU(),

            fc11=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)

        h = self.prelu1(self.bn1(self.conv1(x)))
        h = self.prelu2(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.prelu3(self.bn3(self.conv3(h)))
        h = self.prelu4(self.bn4(self.conv4(h)))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.prelu5(self.bn5(self.conv5(h)))
        h = self.prelu6(self.bn6(self.conv6(h)))
        h = self.prelu7(self.bn7(self.conv7(h)))
        h = self.prelu8(self.bn8(self.conv8(h)))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(self.prelu9(self.fc9(h)), train=train, ratio=0.5)
        h = F.dropout(self.prelu10(self.fc10(h)), train=train, ratio=0.5)
        h = self.fc11(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGG_mini_ABN_PReLU(FunctionSet):

    """

    VGGnet for CIFAR-10

    """

    def __init__(self):
        super(VGG_mini_ABN_PReLU, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1_1=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu1_1=F.PReLU(),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1_2=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu1_2=F.PReLU(),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn2_1=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu2_1=F.PReLU(),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2_2=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu2_2=F.PReLU(),

            conv3_1=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn3_1=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            prelu3_1=F.PReLU(),
            conv3_2=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_2=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            prelu3_2=F.PReLU(),
            conv3_3=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_3=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            prelu3_3=F.PReLU(),
            conv3_4=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_4=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            prelu3_4=F.PReLU(),

            fc4=F.Linear(4096, 1024),
            prelu4=F.PReLU(),

            fc5=F.Linear(1024, 1024),
            prelu5=F.PReLU(),

            fc6=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = self.prelu1_1(self.bn1_1(self.conv1_1(x)))
        h = self.prelu1_2(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = self.prelu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.prelu2_2(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = self.prelu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.prelu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.prelu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.prelu3_4(self.bn3_4(self.conv3_4(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = F.dropout(self.prelu4(self.fc4(h)), train=train, ratio=0.5)
        h = F.dropout(self.prelu5(self.fc5(h)), train=train, ratio=0.5)
        h = self.fc6(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

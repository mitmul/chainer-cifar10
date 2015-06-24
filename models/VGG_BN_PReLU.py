#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGG_BN_PReLU(FunctionSet):

    """

    VGGnet withParameterized ReLU

    """

    def __init__(self):
        super(VGG_BN_PReLU, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1_1=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu1_1=F.PReLU(),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1_2=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            prelu1_2=F.PReLU(),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn2_1=F.BatchNormalization(128, decay=0.9, eps=1e-5),
            prelu2_1=F.PReLU(),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2_2=F.BatchNormalization(128, decay=0.9, eps=1e-5),
            prelu2_2=F.PReLU(),

            conv3_1=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            prelu3_1=F.PReLU(),
            conv3_2=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            prelu3_2=F.PReLU(),
            conv3_3=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            prelu3_3=F.PReLU(),

            conv4_1=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            prelu4_1=F.PReLU(),
            conv4_2=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            prelu4_2=F.PReLU(),
            conv4_3=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            prelu4_3=F.PReLU(),

            conv5_1=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            prelu5_1=F.PReLU(),
            conv5_2=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            prelu5_2=F.PReLU(),
            conv5_3=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            prelu5_3=F.PReLU(),

            fc6=F.Linear(4608, 4096),
            prelu6=F.PReLU(),
            fc7=F.Linear(4096, 4096),
            prelu7=F.PReLU(),
            fc8=F.Linear(4096, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = self.prelu1_1(self.bn1_1(self.conv1_1(x)))
        h = self.prelu1_2(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.prelu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.prelu2_2(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.prelu3_1(self.conv3_1(h))
        h = self.prelu3_2(self.conv3_2(h))
        h = self.prelu3_3(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = self.prelu4_1(self.conv4_1(h))
        h = self.prelu4_2(self.conv4_2(h))
        h = self.prelu4_3(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = self.prelu5_1(self.conv5_1(h))
        h = self.prelu5_2(self.conv5_2(h))
        h = self.prelu5_3(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = F.dropout(self.prelu6(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(self.prelu7(self.fc7(h)), train=train, ratio=0.5)
        h = self.fc8(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

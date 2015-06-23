#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGG_BN(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(VGG_BN, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1=F.BatchNormalization(64),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2=F.BatchNormalization(128),

            conv3_1=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=F.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv4_4=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_5=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_6=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc5=F.Linear(25088, 4096),
            fc6=F.Linear(4096, 4096),
            fc7=F.Linear(4096, 28)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.bn1(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.bn2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        h = self.fc8(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

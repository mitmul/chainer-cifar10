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
            conv2=F.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv3=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv4=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv5=F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv6=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv7=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv8=F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv9=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv10=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv11=F.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv12=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv13=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv14=F.Convolution2D(256, 256, 3, stride=1, pad=1),

            fc15=F.Linear(1024, 1024),
            fc16=F.Linear(1024, 1024),
            pred=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = F.relu(self.conv12(h))
        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.max_pooling_2d(h, 2, stride=1)

        h = F.dropout(F.relu(self.fc15(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc16(h)), train=train, ratio=0.5)
        h = self.pred(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

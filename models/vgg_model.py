#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGGNet(FunctionSet):

    def __init__(self):
        super(VGGNet, self).__init__(
            conv1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1=F.BatchNormalization(64),

            conv2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn2=F.BatchNormalization(64),

            conv3=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn3=F.BatchNormalization(128),

            conv4=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn4=F.BatchNormalization(128),

            conv5=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn5=F.BatchNormalization(256),

            conv6=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn6=F.BatchNormalization(256),

            conv7=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn7=F.BatchNormalization(256),

            conv8=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn8=F.BatchNormalization(256),

            fc9=F.Linear(4096, 1024),
            fc10=F.Linear(1024, 1024),
            fc11=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.dropout(F.max_pooling_2d(h, 2, stride=2), ratio=0.25)

        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.dropout(F.max_pooling_2d(h, 2, stride=2), ratio=0.25)

        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.relu(self.bn7(self.conv7(h)))
        h = F.relu(self.bn8(self.conv8(h)))
        h = F.dropout(F.max_pooling_2d(h, 2, stride=2), ratio=0.25)

        h = F.dropout(F.relu(self.fc9(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc10(h)), train=train, ratio=0.5)
        h = self.fc11(h)

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

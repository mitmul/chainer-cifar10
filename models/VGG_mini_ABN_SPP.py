#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F


class VGG_mini_ABN_SPP(FunctionSet):

    """

    VGGnet for CIFAR-10

    """

    def __init__(self):
        super(VGG_mini_ABN_SPP, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            bn1_1=F.BatchNormalization(64, decay=0.9, eps=1e-5),
            conv1_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
            bn1_2=F.BatchNormalization(64, decay=0.9, eps=1e-5),

            conv2_1=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            bn2_1=F.BatchNormalization(128, decay=0.9, eps=1e-5),
            conv2_2=F.Convolution2D(128, 128, 3, stride=1, pad=1),
            bn2_2=F.BatchNormalization(128, decay=0.9, eps=1e-5),

            conv3_1=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            bn3_1=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            conv3_2=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_2=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            conv3_3=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_3=F.BatchNormalization(256, decay=0.9, eps=1e-5),
            conv3_4=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            bn3_4=F.BatchNormalization(256, decay=0.9, eps=1e-5),

            fc4=F.Linear(5376, 1024),
            fc5=F.Linear(1024, 1024),
            fc6=F.Linear(1024, 10)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(h, ratio=0.25, train=train)

        h = F.dropout(F.relu(self.fc4(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc5(h)), train=train, ratio=0.5)
        h = self.fc6(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h

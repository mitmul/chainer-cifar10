import chainer
import chainer.functions as F
import chainer.links as L


class VGG(chainer.Chain):

    def __init__(self, n_class=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Linear(None, 1024)
            self.fc5 = L.Linear(1024, 1024)
            self.fc6 = L.Linear(1024, n_class)

    def __call__(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5)
        h = self.fc6(h)
        return h

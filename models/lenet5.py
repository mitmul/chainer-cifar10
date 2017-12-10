import chainer
import chainer.functions as F
import chainer.links as L


class LeNet5(chainer.Chain):

    def __init__(self, n_class=10):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 5, stride=1, pad=2)
            self.conv2 = L.Convolution2D(32, 32, 5, stride=1, pad=2)
            self.conv3 = L.Convolution2D(32, 64, 5, stride=1, pad=2)
            self.fc4 = L.Linear(None, 4096)
            self.fc5 = L.Linear(4096, n_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = self.fc5(h)
        return h

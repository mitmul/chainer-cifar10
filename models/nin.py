import chainer
import chainer.functions as F
import chainer.links as L


class NIN(chainer.Chain):

    def __init__(self, n_class=10):
        w = chainer.initializers.HeNormal()
        super(NIN, self).__init__()
        with self.init_scope():
            self.mlpconv1 = L.MLPConvolution2D(
                3, (192, 160, 96), 5, pad=2, conv_init=w)
            self.mlpconv2 = L.MLPConvolution2D(
                96, (192, 192, 192), 5, pad=2, conv_init=w)
            self.mlpconv3 = L.MLPConvolution2D(
                192, (192, 192, n_class), 3, pad=1, conv_init=w)

    def __call__(self, x):
        h = F.relu(self.mlpconv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5)

        h = F.relu(self.mlpconv2(h))
        h = F.average_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, ratio=0.5)

        h = self.mlpconv3(h)
        h = F.average_pooling_2d(h, h.shape[2])
        y = F.reshape(h, (x.shape[0], 10))
        return y


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = NIN(10)
    y = model(x)

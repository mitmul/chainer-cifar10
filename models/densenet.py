import chainer
import chainer.functions as F
import chainer.links as L


class BNReLUConvDropConcat(chainer.Chain):

    def __init__(self, in_ch, out_ch, dropout_ratio):
        super(BNReLUConvDropConcat, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = self.conv(F.relu(self.bn(x)))
        if self.dropout_ratio > 0:
            h = F.dropout(h, ratio=self.dropout_ratio)
        return F.concat((x, h))


class DenseBlock(chainer.ChainList):

    def __init__(self, in_ch, growth_rate, n_layer, dropout_ratio):
        super(DenseBlock, self).__init__()
        for i in range(n_layer):
            self.add_link(BNReLUConvDropConcat(
                in_ch + i * growth_rate, growth_rate, dropout_ratio))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class Transition(chainer.Chain):

    def __init__(self, in_ch, dropout_ratio):
        super(Transition, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, in_ch, 1, 1, 0, initialW=w)
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = F.relu(self.bn(x))
        if self.dropout_ratio > 0:
            h = F.dropout(self.conv(h), ratio=self.dropout_ratio)
        h = F.average_pooling_2d(h, 2)
        return h


class BNReLUAPoolFC(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        super(BNReLUAPoolFC, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.fc = L.Linear(in_ch, out_ch)

    def __call__(self, x):
        h = F.relu(self.bn(x))
        h = F.average_pooling_2d(h, h.shape[2:])
        return self.fc(h)


class DenseNet(chainer.ChainList):
    def __init__(
            self, n_layer=32, growth_rate=24, n_class=10, dropout_ratio=0,
            in_ch=16, n_block=3):
        super(DenseNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.add_link(L.Convolution2D(None, in_ch, 3, 1, 1, initialW=w))
            for i in range(n_block):
                _in_ch = in_ch + n_layer * growth_rate * i
                self.add_link(DenseBlock(
                    _in_ch, growth_rate, n_layer, dropout_ratio))
                if i < n_block - 1:
                    _in_ch = in_ch + n_layer * growth_rate * (i + 1)
                    self.add_link(Transition(_in_ch, dropout_ratio))
            _in_ch = in_ch + n_layer * growth_rate * n_block
            self.add_link(BNReLUAPoolFC(_in_ch, n_class))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    model = DenseNet(10)
    y = model(x)
    from chainer import computational_graph
    cg = computational_graph.build_computational_graph([y])
    with open('densenet.dot', 'w') as fp:
        fp.write(cg.dump())
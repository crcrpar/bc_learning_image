import math

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.initializers import Uniform

from chainercv.links import Conv2DBNActiv


def kl_divergence(y, t):
    entropy = - F.sum(t[t.nonzero()] * F.log(t[t.nonzero()]))
    cross_entropy = - F.sum(t * F.log_softmax(y))

    return (cross_entropy - entropy) / y.shape[0]


class ConvNet(chainer.Chain):
    """11-layer VGG like Convolutional Neural Network."""

    def __init__(self, n_classes, bc_learning=True, nobias=False, dr_ratio=0.5):
        super(ConvNet, self).__init__()
        self.dr_ratio = dr_ratio
        self.bc_learning = bc_learning
        if self.bc_learning:
            self.loss = kl_divergence
        else:
            self.loss = F.softmax_cross_entropy
        # architecture
        kwargs = {'ksize': 3, 'stride': 1, 'pad': 1, 'nobias': nobias}
        with self.init_scope():
            self.conv1_1 = Conv2DBNActiv(3, 64, **kwargs)
            self.conv1_2 = Conv2DBNActiv(64, 64, **kwargs)
            self.conv2_1 = Conv2DBNActiv(64, 128, **kwargs)
            self.conv2_2 = Conv2DBNActiv(128, 128, **kwargs)
            self.conv3_1 = Conv2DBNActiv(128, 256, **kwargs)
            self.conv3_2 = Conv2DBNActiv(256, 256, **kwargs)
            self.conv3_3 = Conv2DBNActiv(256, 256, **kwargs)
            self.conv3_4 = Conv2DBNActiv(256, 256, **kwargs)
            self.fc4 = L.Linear(1024, initialW=Uniform(
                1. / math.sqrt(256 * 4 * 4)))
            self.fc5 = L.Linear(1024, initialW=Uniform(1. / math.sqrt(1024)))
            self.fc6 = L.Linear(
                n_classes, initialW=Uniform(1. / math.sqrt(1024)))

    def __call__(self, x, t):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, t)
        if chainer.config.train:
            loss = self.loss(y_hat, t)
        else:
            loss = F.softmax_cross_entropy(y_hat, t)
        if self.bc_learning:
            t = F.argmax(t, axis=1)
        acc = F.accuracy(y_hat, t)
        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss

    def forward(self, x):
        # convolve input
        h = self.conv1_2(self.conv1_1(x))
        h = F.max_pooling_2d(h, 2)
        h = self.conv2_2(self.conv2_1(h))
        h = F.max_pooling_2d(h, 2)
        h = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(h))))
        h = F.max_pooling_2d(h, 2)
        # linear
        h = F.dropout(F.relu(self.fc4(h)), self.dr_ratio)
        h = F.dropout(F.relu(self.fc5(h)), self.dr_ratio)
        return self.fc6(h)

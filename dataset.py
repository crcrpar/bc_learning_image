import numpy as np

import chainer
from chainer import iterators
import chainercv.transforms as T


def _load_cifar(cifar10, train):
    if cifar10:
        train, test = chainer.datasets.get_cifar10(scale=255.)
    else:
        train, test = chainer.datasets.get_cifar100(scale=255.)
    if train:
        return train
    else:
        return test


def normalize(image, mean, std):
    return (image - mean[:, None, None]) / std[:, None, None]


def zero_mean(image, mean, std):
    img_mean = np.mean(image, keepdims=True)
    return (image - img_mean - mean[:, None, None]) / std[:, None, None]


def padding(image, pad):
    return np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant')


class ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, opt, cifar10=True, train=True):
        self.opt = opt
        self.cifar10 = cifar10
        self.train = train
        self.base = _load_cifar(self.cifar10, self.train)
        self.mix = opt.BC and train
        if opt.dataset == 'cifar10':
            if opt.plus:
                self.mean = np.array([4.60, 2.24, -6.84])
                self.std = np.array([55.9, 53.7, 56.5])
            else:
                self.mean = np.array([125.3, 123.0, 113.9])
                self.std = np.array([63.0, 62.1, 66.7])
        else:
            if opt.plus:
                self.mean = np.array([7.37, 2.13, -9.50])
                self.std = np.array([57.6, 54.0, 58.5])
            else:
                self.mean = np.array([129.3, 124.1, 112.4])
                self.std = np.array([68.2, 65.4, 70.4])

        self.N = len(self.base)
        if self.opt.plus:
            self.normalize = zero_mean
        else:
            self.normalize = normalize

    def __len__(self):
        return self.N

    def preprocess(self, image):
        image = self.normalize(image, self.mean, self.std)
        if not self.train:
            return image
        else:
            image = T.random_flip(image, x_random=True)
            image = padding(image, 4)
            image = T.random_crop(image, 32)
            return image

    def get_example(self, i):
        if self.mix:
            while True:
                i1, i2 = np.random.randint(0, self.N, 2)
                image1, label1 = self.base[i1]
                image2, label2 = self.base[i2]
                if label1 != label2:
                    break
            image1 = self.preprocess(image1)
            image2 = self.preprocess(image2)
            r = np.random.rand(1)
            if self.opt.plus:
                g1 = np.std(image1)
                g2 = np.std(image2)
                p = 1. / (1 + g1 / g2 * (1 - r) / r)
                image = ((image1 * p + image2 * (1 - p)) /
                         np.sqrt(p ** 2 + (1 - p) ** 2)).astype(np.float32)
            else:
                image = (image1 * p + image2 * (1 - r)).astype(np.float32)

            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2]
                     * (1 - r)).astype(np.float32)

        else:
            image, label = self.base[i]
            image = self.preprocess(image).astype(np.float32)
            label = np.asarray(label, dtype=np.int32)

        return image, label


def setup(opt):
    # Iterator setup
    cifar10 = opt.dataset.lower() == 'cifar10'
    train_data = ImageDataset(opt, cifar10, train=True)
    val_data = ImageDataset(opt, cifar10, train=False)
    train_iter = iterators.MultiprocessIterator(
        train_data, opt.batchSize, repeat=False)
    val_iter = iterators.SerialIterator(
        val_data, opt.batchSize, repeat=False, shuffle=False)

    return train_iter, val_iter

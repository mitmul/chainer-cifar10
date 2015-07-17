#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import shift


class Transform(object):

    def __init__(self, **params):
        [setattr(self, key, value) for key, value in params.items()]

    def transform(self, img):
        self._img = img
        if hasattr(self, 'flip'):
            self.fliplr()
        if hasattr(self, 'shift'):
            self.translate()
        if hasattr(self, 'size'):
            if not isinstance(self.size, tuple):
                self.size = (self.size, self.size)
            self.scale()
        if hasattr(self, 'norm'):
            if self.norm:
                if not self._img.dtype == np.float32:
                    self._img = self._img.astype(np.float32)
                # global contrast normalization
                for ch in range(self._img.shape[2]):
                    im = self._img[:, :, ch]
                    im = (im - np.mean(im)) / \
                        (np.std(im) + np.finfo(np.float32).eps)
                    self._img[:, :, ch] = im

        if not self._img.dtype == np.float32:
            self._img = self._img.astype(np.float32)

        return self._img

    def fliplr(self):
        if np.random.randint(2) == 1 and self.flip == True:
            self._img = np.fliplr(self._img)

    def translate(self):
        dx = int(np.random.rand() * self.shift * 2 - self.shift)
        dy = int(np.random.rand() * self.shift * 2 - self.shift)
        self._img = shift(self._img, (dy, dx, 0))

        if dx < 0:
            self._img = self._img[:, :dx, :]
        if dx >= 0:
            self._img = self._img[:, dx:, :]

        if dy < 0:
            self._img = self._img[:dy, :, :]
        if dy >= 0:
            self._img = self._img[dy:, :, :]

    def scale(self):
        self._img = imresize(self._img, self.size, 'nearest')


if __name__ == '__main__':
    train_data = np.load('data/train_data.npy')
    trans = Transform(angle=5, flip=True, shift=5, size=(32, 32))

    for i in range(10):
        img = train_data[i].transpose((1, 2, 0)) * 255
        img = img.astype(np.uint8)[:, :, ::-1]
        img = trans.transform(img)
        cv.imshow('test', img)
        cv.waitKey(0)

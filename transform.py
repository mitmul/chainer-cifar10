#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from skimage.io import imsave
from scipy.misc import imresize
from scipy.ndimage.interpolation import shift


class Transform(object):

    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        self.img = img

        if self.args.flip == 1:
            if np.random.randint(2) == 0:
                self.img = np.fliplr(self.img)

        if self.args.shift > 0:
            dx = int(np.random.rand() * self.args.shift * 2 - self.args.shift)
            dy = int(np.random.rand() * self.args.shift * 2 - self.args.shift)
            self.img = shift(self.img, (dy, dx, 0))

            if dx < 0:
                self.img = self.img[:, :dx, :]
            if dx >= 0:
                self.img = self.img[:, dx:, :]

            if dy < 0:
                self.img = self.img[:dy, :, :]
            if dy >= 0:
                self.img = self.img[dy:, :, :]

        if self.args.crop > 0:
            size = (self.args.crop, self.args.crop)
            self.img = imresize(self.img, size, 'nearest')

        if not self.img.dtype == np.float32:
            self.img = self.img.astype(np.float32)

        if self.args.norm == 1:
            self.img -= self.img.reshape(-1, 3).mean(axis=0)
            self.img /= self.img.reshape(-1, 3).std(axis=0) + 1e-5

        return self.img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--shift', type=int, default=10)
    parser.add_argument('--crop', type=int, default=28)
    parser.add_argument('--norm', type=int, default=0)
    args = parser.parse_args()
    trans = Transform(args)

    train_data = np.load('data/train_data.npy')
    if not os.path.exists('data/test_trans'):
        os.mkdir('data/test_trans')
    for i in range(10):
        img = train_data[i]
        img = trans(img)
        imsave('data/test_trans/{}.png'.format(i), img)

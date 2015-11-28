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
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--shift', type=int, default=10)
    parser.add_argument('--crop', type=int, default=28)
    parser.add_argument('--norm', type=int, default=0)
    args = parser.parse_args()
    trans = Transform(args)

    data = np.load('data/train_data.npy')
    labels = np.load('data/train_labels.npy')
    perm = np.random.permutation(data.shape[0])
    if not os.path.exists('data/test_trans'):
        os.mkdir('data/test_trans')
    for i in range(0, data.shape[0], args.batchsize):
        chosen_ids = perm[i:i + args.batchsize]
        img = data[chosen_ids]
        lbl = labels[chosen_ids]
        aug = np.empty((len(chosen_ids), args.crop, args.crop, 3),
                       dtype=np.float32)
        for j, k in enumerate(chosen_ids):
            aug[j] = trans(data[k])

        for im, lb in zip(aug, lbl):
            imsave('data/test_trans/{}-{}_{}_{}.png'.format(
                lb, i, j, k), im.astype(np.uint8))

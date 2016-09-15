#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy.misc import imresize
from skimage.io import imsave

import argparse
import numpy as np
import os
import six


class Transform(object):

    cropping_size = 24
    scaling_size = 28

    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        imgs = []

        for offset_y in six.moves.range(0, 8 + 4, 4):
            for offset_x in six.moves.range(0, 8 + 4, 4):
                im = img[offset_y:offset_y + self.cropping_size,
                         offset_x:offset_x + self.cropping_size]
                # global contrast normalization
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im /= im.reshape(-1, 3).std(axis=0) + 1e-5

                imgs.append(im)
                imgs.append(np.fliplr(im))

        for offset_y in six.moves.range(0, 4 + 2, 2):
            for offset_x in six.moves.range(0, 4 + 2, 2):
                im = img[offset_y:offset_y + self.scaling_size,
                         offset_x:offset_x + self.scaling_size]
                im = imresize(im, (self.cropping_size, self.cropping_size),
                              'nearest')
                # global contrast normalization
                im = im.astype(np.float)
                im -= im.reshape(-1, 3).mean(axis=0)
                im /= im.reshape(-1, 3).std(axis=0) + 1e-5

                imgs.append(im)
                imgs.append(np.fliplr(im))
        imgs = np.asarray(imgs, dtype=np.float32)

        return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=int, default=1)
    args = parser.parse_args()
    trans = Transform(args)
    np.random.seed(1701)

    data = np.load('data/train_data.npy')
    labels = np.load('data/train_labels.npy')
    perm = np.random.permutation(data.shape[0])
    if not os.path.exists('data/test_trans'):
        os.mkdir('data/test_trans')
    for i in range(0, data.shape[0], args.batchsize):
        chosen_ids = perm[i:i + args.batchsize]
        img = data[chosen_ids]
        lbl = labels[chosen_ids]
        aug = np.empty((len(chosen_ids), 24, 24, 3), dtype=np.float32)
        for j, k in enumerate(chosen_ids):
            aug[j] = trans(data[k])
            d = aug[j]
            d -= d.min()
            d /= d.max()
            aug[j] *= 255

        for im, lb in zip(aug, lbl):
            imsave('data/test_trans/{}-{}_{}_{}.png'.format(
                lb, i, j, k), im.astype(np.uint8))

        print(i)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
from skimage.io import imsave
from six.moves import cPickle as pickle


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


def load_dataset(datadir='data'):
    train_data = np.load('%s/train_data.npy' % datadir)
    train_labels = np.load('%s/train_labels.npy' % datadir)
    test_data = np.load('%s/test_data.npy' % datadir)
    test_labels = np.load('%s/test_labels.npy' % datadir)

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', type=str, default='data')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    data = np.zeros((50000, 3, 32, 32), dtype=np.uint8)
    labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob('cifar-10-batches-py/data_batch*'))):
        print(data_fn)
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = \
            batch['data'].reshape((10000, 3, 32, 32))
        labels.extend(batch['labels'])
    data = data.transpose((0, 2, 3, 1))
    labels = np.asarray(labels, dtype=np.int32)

    if not os.path.exists('data/test_data'):
        os.mkdir('data/test_data')
    for i in range(50000):
        imsave('data/test_data/{}.png'.format(i), data[i])

    np.save('%s/train_data' % args.outdir, data)
    np.save('%s/train_labels' % args.outdir, labels)

    test = unpickle('cifar-10-batches-py/test_batch')

    data = np.asarray(test['data'], dtype=np.uint8).reshape(
        (10000, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = np.asarray(test['labels'], dtype=np.int32)
    for i in range(100, 200, 1):
        imsave('data/test_data/{}.png'.format(i), data[i])

    np.save('%s/test_data' % args.outdir, data)
    np.save('%s/test_labels' % args.outdir, labels)

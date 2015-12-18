#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import numpy as np
from skimage.io import imsave
from six.moves import cPickle as pickle
from scipy import linalg
from transform import Transform


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


def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data')
    parser.add_argument('--whitening', type=int, default=1)
    parser.add_argument('--norm', type=int, default=1)
    args = parser.parse_args()
    print(args)

    trans = Transform(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # prepare training dataset
    data = np.zeros((50000, 3 * 32 * 32), dtype=np.float)
    labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob('cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels.extend(batch['labels'])
    if args.whitening == 1:
        components, mean, data = preprocessing(data)

    data = data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = np.asarray(labels, dtype=np.int32)
    training_data = []
    training_labels = []
    for d, l in zip(data, labels):
        imgs = trans(d)
        for img in imgs:
            training_data.append(img)
            training_labels.append(l)
    training_data = np.array(training_data, dtype=np.float32)
    training_labell = np.array(training_labels, dtype=np.int32)
    np.save('%s/train_data' % args.outdir, training_data)
    np.save('%s/train_labels' % args.outdir, training_labels)

    # saving training dataset
    if not os.path.exists('data/test_data'):
        os.mkdir('data/test_data')
    for i in range(50000):
        d = data[i]
        d -= d.min()
        d /= d.max()
        d = (d * 255).astype(np.uint8)
        imsave('data/test_data/train_{}.png'.format(i), d)

    test = unpickle('cifar-10-batches-py/test_batch')
    data = np.asarray(test['data'], dtype=np.float)
    if args.whitening == 1:
        mdata = data - mean
        data = np.dot(mdata, components.T)
    data = data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = np.asarray(test['labels'], dtype=np.int32)
    np.save('%s/test_data' % args.outdir, data)
    np.save('%s/test_labels' % args.outdir, labels)

    for i in range(10000):
        d = data[i]
        d -= d.min()
        d /= d.max()
        d = (d * 255).astype(np.uint8)
        imsave('data/test_data/test_{}.png'.format(i), d)

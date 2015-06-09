#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import cPickle
import argparse


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()

    return dict


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

    data = []
    labels = []
    for data_fn in sorted(glob.glob('cifar-10-batches-py/data_batch*')):
        batch = unpickle(data_fn)
        data.append(batch['data'])
        labels.append(batch['labels'])

    data = np.asarray(data)
    _data = data[0]
    for d in data[1:]:
        _data = np.vstack((_data, d))
    data = _data

    labels = np.asarray(labels)
    labels = labels.reshape((labels.shape[0], labels.shape[1], 1))
    _labels = labels[0]
    for l in labels[1:]:
        _labels = np.vstack((_labels, l))
    labels = _labels.reshape((labels.shape[0] * labels.shape[1]))

    num, dim = data.shape
    train_data = data.reshape((num, 3, 32, 32)).astype(np.float32) / 255.0
    train_labels = labels.astype(np.int32)

    np.save('%s/train_data' % args.outdir, train_data)
    np.save('%s/train_labels' % args.outdir, train_labels)

    test = unpickle('cifar-10-batches-py/test_batch')

    data = np.asarray(test['data'])
    num, dim = data.shape
    test_data = data.reshape((num, 3, 32, 32)).astype(np.float32) / 255.0
    test_labels = np.asarray(test['labels'], dtype=np.int32)

    np.save('%s/test_data' % args.outdir, test_data)
    np.save('%s/test_labels' % args.outdir, test_labels)

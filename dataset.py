#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import cPickle


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()

    return dict


def load_dataset():
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
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

    np.save('train_data', train_data)
    np.save('train_labels', train_labels)

    test = unpickle('cifar-10-batches-py/test_batch')
    data = test['data']
    labels = np.asarray(test['labels'])
    # labels = labels.reshape((labels.shape[0], 1))
    num, dim = data.shape
    test_data = data.reshape((num, 3, 32, 32)).astype(np.float32) / 255.0

    np.save('test_data', test_data)
    np.save('test_labels', labels)

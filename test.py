#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import imp
from chainer import cuda, Variable
import chainer.functions as F
from dataset import load_dataset
from transform import Transform
import cPickle as pickle
import numpy as np
import cv2 as cv
import argparse


def single_eval(test_data, test_labels, N_test, model, gpu=0):
    n_dup = 1
    sum_accuracy = 0
    for i in xrange(N_test):
        single_x = test_data[i]
        single_y = test_labels[i]

        single_x = np.tile(single_x, (n_dup, 1, 1, 1)).astype(np.float32)
        single_y = np.tile(single_y, (n_dup,)).astype(np.int32)

        if gpu >= 0:
            single_x = cuda.to_gpu(single_x)
            single_y = cuda.to_gpu(single_y)

        _, _, pred = model.forward(single_x, single_y, train=False)
        pred = cuda.to_cpu(F.softmax(pred).data)[0]

        pred_class = np.argmax(pred)
        true_class = test_labels[i]

        if pred_class == true_class:
            sum_accuracy += 1
        else:
            if not os.path.exists('%d' % test_labels[i]):
                os.mkdir('%d' % test_labels[i])

            cv.imwrite('%d/%d-%d.jpg' % (test_labels[i], pred_class, i),
                       test_data[i].transpose((1, 2, 0)) * 255)

        if i % 100 == 0:
            print i, N_test, sum_accuracy / float(i + 1)

    return sum_accuracy


def aug_eval(test_data, test_labels, N_test, model, gpu=0):
    trans = Transform(angle=15,
                      flip=True,
                      shift=10,
                      size=(32, 32),
                      norm=False)

    # evaluation
    n_dup = 64
    sum_accuracy = 0
    for i in xrange(N_test):
        single_x = test_data[i]
        single_y = test_labels[i]

        aug_x = np.tile(single_x, (n_dup, 1, 1, 1)).astype(np.float32)
        aug_y = np.tile(single_y, (n_dup,)).astype(np.int32)

        # data augmentation
        for j in range(n_dup):
            aug_x[j] = trans.transform(
                single_x.transpose((1, 2, 0))).transpose((2, 0, 1))

        if gpu >= 0:
            aug_x = cuda.to_gpu(aug_x)
            aug_y = cuda.to_gpu(aug_y)

        _, _, pred = model.forward(aug_x, aug_y, train=False)
        mean_pred = cuda.to_cpu(F.softmax(pred).data)

        mean_pred = np.sum(mean_pred, axis=0)
        pred = np.argmax(mean_pred)
        true = cuda.to_cpu(aug_y)[0]

        if pred == true:
            sum_accuracy += 1

        if i % 100 == 0:
            print i, n_dup, N_test, sum_accuracy / float(i + 1)

    print sum_correct / float(N_test)

    return sum_accuracy


def eval(test_data, test_labels, N_test, batchsize, model, gpu):
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    sum_correct = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = test_data[i:i + batchsize]
        y_batch = test_labels[i:i + batchsize]

        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc, pred = model.forward(x_batch, y_batch, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        pred = cuda.to_cpu(pred.data).argmax(axis=1)
        labels = test_labels[i:i + batchsize]

        sum_correct += np.sum(pred == labels)

    print sum_correct / float(N_test)

    return sum_loss, sum_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', '-e', type=str, default='normal')
    args = parser.parse_args()
    print(args)

    cuda.init()
    module = imp.load_source('Vgg', 'results/vgg_aug_1/vgg_model.py')
    model = pickle.load(
        open('results/vgg_aug_1/vgg_aug_1_epoch_50.chainermodel', 'rb'))

    _, _, test_data, test_labels = load_dataset()
    N_test = test_data.shape[0]
    batchsize = 128
    gpu = 0

    if args.eval == 'normal':
        sum_loss, sum_accuracy = eval(
            test_data, test_labels, N_test, batchsize, model, gpu)
        print('test mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    if args.eval == 'aug':
        sum_accuracy = aug_eval(
            test_data, test_labels, N_test, model, gpu)
        print('test aug mean accuracy={}'.format(
            sum_accuracy / float(N_test)))

    if args.eval == 'single':
        sum_accuracy = single_eval(
            test_data, test_labels, N_test, model, gpu)
        print('test aug mean accuracy={}'.format(sum_accuracy / float(N_test)))

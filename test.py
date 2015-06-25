#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import imp
from chainer import cuda
import chainer.functions as F
from dataset import load_dataset
from transform import Transform
import pickle
import numpy as np
import argparse
from train import norm
from progressbar import ProgressBar


def aug_eval(test_data, test_labels, N_test, model, gpu=0):
    trans = Transform(norm=True)
    # evaluation
    n_dup = 1
    sum_correct = 0
    pbar = ProgressBar(N_test)
    for i in range(N_test):
        single_x = test_data[i]
        single_y = test_labels[i]

        # create a set of the same image
        aug_x = np.tile(single_x, (n_dup, 1, 1, 1)).astype(np.float32)
        aug_y = np.tile(single_y, (n_dup,)).astype(np.int32)

        # data augmentation
        for j in range(n_dup):
            x = aug_x[j].transpose((1, 2, 0))
            x = trans.transform(x).transpose((2, 0, 1))
            aug_x[j] = x

        if gpu >= 0:
            aug_x = cuda.to_gpu(aug_x)
            aug_y = cuda.to_gpu(aug_y)

        _, _, pred = model.forward(aug_x, aug_y, train=False)
        mean_pred = cuda.to_cpu(F.softmax(pred).data)
        mean_pred = np.mean(mean_pred, axis=0)
        pred = np.argmax(mean_pred)
        true = cuda.to_cpu(aug_y)[0]
        sys.exit()

        if pred == true:
            sum_correct += 1

        if i % 100 == 0:
            print(i, n_dup, N_test, sum_correct / float(i + 1))

        pbar.update(i + args.batchsize
                    if (i + args.batchsize) < N_test else N_test)

    print(sum_correct / float(N_test))

    return sum_accuracy


def validate(test_data, test_labels, N_test, model, args):
    # validate
    pbar = ProgressBar(N_test)
    sum_accuracy = 0
    sum_loss = 0
    sum_correct = 0
    for i in range(0, N_test, args.batchsize):
        x_batch = test_data[i:i + args.batchsize]
        y_batch = test_labels[i:i + args.batchsize]

        if args.norm == 1:
            x_batch = np.asarray(map(norm, x_batch))

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32))
            y_batch = cuda.to_gpu(y_batch.astype(np.int32))

        loss, acc, pred = model.forward(x_batch, y_batch, train=False)

        pred = cuda.to_cpu(F.softmax(pred).data)
        pred = np.argmax(pred, axis=1)
        label = cuda.to_cpu(y_batch)
        sum_correct += np.sum(np.array(pred == label))

        sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * args.batchsize
        pbar.update(i + batchsize if (i + batchsize) < N_test else N_test)

    print('correct num:{}\t# of test images:{}'.format(sum_correct, N_test))
    print('correct rate:{}'.format(float(sum_correct) / float(N_test)))

    return sum_loss, sum_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, default='normal',
                        choices=['normal', 'aug', 'single'])
    parser.add_argument('--model', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--norm', type=int)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.init()
    model_n = os.path.basename(args.model).split('.')[0]
    module = imp.load_source(model_n, args.model)
    model = pickle.load(open(args.param, 'rb'), encoding='latin1')
    if args.gpu >= 0:
        model.to_gpu()
    else:
        model.to_cpu()

    _, _, test_data, test_labels = load_dataset()
    N_test = test_data.shape[0]
    batchsize = 128

    if args.eval == 'normal':
        sum_loss, sum_accuracy = validate(
            test_data, test_labels, N_test, model, args)
        print('test mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    if args.eval == 'aug':
        sum_accuracy = aug_eval(
            test_data, test_labels, N_test, model, args)
        print('test aug mean accuracy={}'.format(
            sum_accuracy / float(N_test)))

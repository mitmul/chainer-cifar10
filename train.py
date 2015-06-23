#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import logging
import time
import os
import sys
import imp
import shutil
import numpy as np
from chainer import optimizers, cuda, Variable
import chainer.functions as F
from dataset import load_dataset
from transform import Transform
import cPickle as pickle
from draw_loss import draw_loss_curve
from progressbar import ProgressBar


def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = '.'
        log_fn = 'log.txt'
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir

def get_model_optimizer(result_dir, args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.model)
    Net = getattr(module, model_name)

    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(args.model, dst)

    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    model = Net()
    if args.restart_from is not None:
        if args.gpu >= 0:
            cuda.init(args.gpu)
        model = pickle.load(open(args.restart_from, 'rb'))
    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()

    # prepare optimizer
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model.collect_parameters())

    return model, optimizer


def train(train_data, train_labels, N, batchsize, model, optimizer, trans, gpu):
    # training
    pbar = ProgressBar(N)
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = train_data[perm[i:i + batchsize]]
        y_batch = train_labels[perm[i:i + batchsize]]
        # data augmentation
        aug_x = []
        for x in x_batch:
            aug_x.append(
                trans.transform(x.transpose((1, 2, 0))).transpose((2, 0, 1)))
        aug_x = np.asarray(aug_x, dtype=np.float32)
        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch, train=True)
        loss.backward()
        optimizer.weight_decay(decay=0.0005)
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
        pbar.update(i + batchsize if (i + batchsize) < N else N)

    return sum_loss, sum_accuracy


def eval(test_data, test_labels, N_test, batchsize, model, gpu):
    # evaluation
    pbar = ProgressBar(N_test)
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = test_data[i:i + batchsize]
        y_batch = test_labels[i:i + batchsize]

        if gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc, _ = model.forward(x_batch, y_batch, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize
        pbar.update(i + batchsize if (i + batchsize) < N_test else N_test)

    return sum_loss, sum_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='vgg')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--batchsize', '-b', type=int, default=128)
    parser.add_argument('--prefix', '-p', type=str)
    parser.add_argument('--snapshot', '-s', type=int, default=10)
    parser.add_argument('--restart_from', '-r', type=str)
    parser.add_argument('--epoch_offset', '-o', type=int, default=0)
    parser.add_argument('--datadir', '-d', type=str, default='data')
    args = parser.parse_args()

    # create result dir
    log_fn, result_dir = create_result_dir(args)
    logging.basicConfig(filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    model, optimizer = get_model_optimizer(result_dir, args.model, args.gpu)
    train_data, train_labels, test_data, test_labels = \
        load_dataset(args.datadir)
    N = train_data.shape[0]
    N_test = test_data.shape[0]

    # augmentation setting
    trans = Transform(flip=True,
                      shift=5,
                      size=(32, 32),
                      norm=True)

    logging.info('start training...')

    # learning loop
    n_epoch = args.epoch
    batchsize = args.batchsize
    for epoch in range(1, n_epoch + 1):
        # train
        if epoch % 10 == 0:
            optimizer.lr *= 0.1

        sum_loss, sum_accuracy = train(
            train_data, train_labels, N, batchsize, model,
            optimizer, trans, args.gpu)
        logging.info('epoch:{:02d}\ttrain mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N, sum_accuracy / N))

        # eval
        sum_loss, sum_accuracy = eval(
            test_data, test_labels, N_test, batchsize, model, args.gpu)
        logging.info('epoch:{:02d}\ttest mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N_test, sum_accuracy / N_test))

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.prefix, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)

        draw_loss_curve(log_fn, '%s/log.jpg' % result_dir)
        print('learning rate:', optimizer.lr)

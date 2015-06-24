#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('../../')
import argparse
import logging
import time
import os
import imp
import shutil
import numpy as np
from chainer import optimizers, cuda
from dataset import load_dataset
from transform import Transform
import cPickle as pickle
from draw_loss import draw_loss_curve
from progressbar import ProgressBar
from multiprocessing import Process, Queue


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
    if args.opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'Adam':
        optimizer = optimizers.Adam(alpha=0.0001)
    elif args.opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(alpha=args.lr)
    else:
        raise Exception('No optimizer is selected')
    optimizer.setup(model.collect_parameters())

    return model, optimizer


def augmentation(x_batch_queue, aug_x_queue, trans):
    while True:
        x_batch = x_batch_queue.get()
        if x_batch is None:
            break

        aug_x = []
        for x in x_batch:
            aug = trans.transform(x.transpose((1, 2, 0))).transpose((2, 0, 1))
            aug_x.append(aug)
        aug_x_queue.put(np.asarray(aug_x))


def train(train_data, train_labels, N, model, optimizer, trans, args):
    # for parallel augmentation
    x_batch_queue = Queue()
    aug_x_queue = Queue()
    aug_worker = Process(target=augmentation,
                         args=(x_batch_queue, aug_x_queue, trans))
    aug_worker.start()

    # training
    pbar = ProgressBar(N)
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, args.batchsize):
        x_batch = train_data[perm[i:i + args.batchsize]]
        y_batch = train_labels[perm[i:i + args.batchsize]]

        # data augmentation
        x_batch_queue.put(x_batch)
        aug_x = aug_x_queue.get()

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(aug_x.astype(np.float32))
            y_batch = cuda.to_gpu(y_batch.astype(np.int32))

        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch, train=True)
        loss.backward()
        if args.opt in ['AdaGrad', 'MomentumSGD']:
            optimizer.weight_decay(decay=args.decay)
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * args.batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    x_batch_queue.put(None)
    aug_worker.join()

    return sum_loss, sum_accuracy


def norm(x):
    if not x.dtype == np.float32:
        x = x.astype(np.float32)
    # local contrast normalization
    for ch in range(x.shape[2]):
        im = x[:, :, ch]
        im = (im - np.mean(im)) / \
            (np.std(im) + np.finfo(np.float32).eps)
        x[:, :, ch] = im

    return x


def validate(test_data, test_labels, N_test, model, args):
    # validate
    pbar = ProgressBar(N_test)
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N_test, args.batchsize):
        x_batch = test_data[i:i + args.batchsize]
        y_batch = test_labels[i:i + args.batchsize]

        if args.norm:
            x_batch = np.asarray(map(norm, x_batch))

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch.astype(np.float32))
            y_batch = cuda.to_gpu(y_batch.astype(np.int32))

        loss, acc, pred = model.forward(x_batch, y_batch, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * args.batchsize
        pbar.update(i + batchsize if (i + batchsize) < N_test else N_test)

    return sum_loss, sum_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/VGG_mini_BN_PReLU.py')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--prefix', type=str,
                        default='VGG_mini_BN_PReLU_Adam')
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--restart_from', type=str)
    parser.add_argument('--epoch_offset', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--flip', type=bool, default=True)
    parser.add_argument('--shift', type=int, default=10)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--norm', type=bool, default=True)
    parser.add_argument('--opt', type=str, default='Adam',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_freq', type=int, default=100)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    args = parser.parse_args()

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(result_dir, args)
    dataset = load_dataset(args.datadir)
    train_data, train_labels, test_data, test_labels = dataset
    N = train_data.shape[0]
    N_test = test_data.shape[0]

    # augmentation setting
    trans = Transform(flip=args.flip,
                      shift=args.shift,
                      size=(args.size, args.size),
                      norm=args.norm)
    logging.info('start training...')

    # learning loop
    n_epoch = args.epoch
    batchsize = args.batchsize
    for epoch in range(1, n_epoch + 1):
        # train
        if args.opt == 'MomentumSGD':
            print('learning rate:', optimizer.lr)
            if epoch % args.lr_decay_freq == 0:
                optimizer.lr *= args.lr_decay_ratio

        sum_loss, sum_accuracy = train(train_data, train_labels, N,
                                       model, optimizer, trans, args)
        msg = 'epoch:{:02d}\ttrain mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N, sum_accuracy / N)
        logging.info(msg)
        print('\n%s' % msg)

        # validate
        sum_loss, sum_accuracy = validate(test_data, test_labels, N_test,
                                          model, args)
        msg = 'epoch:{:02d}\ttest mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N_test, sum_accuracy / N_test)
        logging.info(msg)
        print('\n%s' % msg)

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.prefix, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)

        draw_loss_curve(log_fn, '%s/log.jpg' % result_dir)

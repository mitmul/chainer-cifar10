#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import time
import os
import imp
import shutil
import chainer
import draw_loss
import numpy as np
from chainer import optimizers, cuda, serializers, Variable
from dataset import load_dataset
from transform import Transform
from multiprocessing import Process, Queue


def create_result_dir(args):
    result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
    result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
    result_dir += str(time.time()).replace('.', '')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    args.log_fn = log_fn
    args.result_dir = result_dir


def get_model_optimizer(args):
    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model

    dst = '%s/%s' % (args.result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(args.model, dst)

    dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # prepare optimizer
    if 'opt' in args:
        # prepare optimizer
        if args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
        elif args.opt == 'Adam':
            optimizer = optimizers.Adam(alpha=args.alpha)
        elif args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
        else:
            raise Exception('No optimizer is selected')

        optimizer.setup(model)

        if args.opt == 'MomentumSGD':
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(args.weight_decay))
        return model, optimizer
    else:
        print('No optimizer generated.')
        return model


def augmentation(args, aug_queue, data, label, train):
    trans = Transform(args)
    perm = np.random.permutation(data.shape[0])
    for i in range(0, data.shape[0], args.batchsize):
        chosen_ids = perm[i:i + args.batchsize]
        if args.crop > 0:
            aug = np.empty((len(chosen_ids), args.crop, args.crop, 3),
                           dtype=np.float32)
        else:
            aug = np.empty((len(chosen_ids), 32, 32, 3), dtype=np.float32)

        if train:
            for j, k in enumerate(chosen_ids):
                aug[j] = trans(data[k])
        elif args.norm == 1:
            for j, k in enumerate(chosen_ids):
                aug[j] = data[k] - data[k].reshape(-1, 3).mean(axis=0)
                aug[j] /= aug[j].reshape(-1, 3).std(axis=0) + 1e-5
        else:
            aug = data[chosen_ids]

        x = np.asarray(aug, dtype=np.float32).transpose((0, 3, 1, 2))
        t = np.asarray(label[chosen_ids], dtype=np.int32)

        aug_queue.put((x, t))


def one_epoch(args, model, optimizer, data, label, epoch, train):
    model.train = train
    xp = cuda.cupy if args.gpu >= 0 else np

    # for parallel augmentation
    aug_queue = Queue()
    aug_worker = Process(target=augmentation,
                         args=(args, aug_queue, data, label, train))
    aug_worker.start()

    sum_accuracy = 0
    sum_loss = 0
    num = 0
    for i in range(0, data.shape[0], args.batchsize):
        x, t = aug_queue.get()
        volatile = 'off' if train else 'on'
        x = Variable(xp.asarray(x), volatile=volatile)
        t = Variable(xp.asarray(t), volatile=volatile)

        if train:
            optimizer.update(model, x, t)
        else:
            model(x, t)

        sum_loss += float(model.loss.data) * t.data.shape[0]
        sum_accuracy += float(model.accuracy.data) * t.data.shape[0]
        num += t.data.shape[0]

        logging.info('iter:{}\t{:05d}/{:05d}\t{}'.format(
            (epoch - 1) + i / data.shape[0], i, data.shape[0],
            sum_accuracy / num))

        del x, t

    if train and (epoch == 1 or epoch % args.snapshot == 0):
        model_fn = '{}/model_epoch-{}.model'.format(args.result_dir, epoch)
        opt_fn = '{}/optimizer_epoch-{}.state'.format(args.result_dir, epoch)
        serializers.save_hdf5(model_fn, model)
        serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info('epoch:{}\ttrain loss:{}\ttrain accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))
    else:
        logging.info('epoch:{}\ttest loss:{}\ttest accuracy:{}'.format(
            epoch, sum_loss / data.shape[0], sum_accuracy / data.shape[0]))

    draw_loss.draw_loss_curve('{}/log.txt'.format(args.result_dir),
                              '{}/log.png'.format(args.result_dir), epoch)

    aug_worker.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/VGG.py')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data')

    # augmentation
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--norm', type=int, default=1)

    # optimization
    parser.add_argument('--opt', type=str, default='Adam',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_freq', type=int, default=100)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1701)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # create result dir
    create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)
    dataset = load_dataset(args.datadir)
    tr_data, tr_labels, te_data, te_labels = dataset

    # learning loop
    for epoch in range(1, args.epoch + 1):
        if args.opt == 'MomentumSGD':
            logging.info('learning rate:{}'.format(optimizer.lr))
            if epoch % args.lr_decay_freq == 0:
                optimizer.lr *= args.lr_decay_ratio

        one_epoch(args, model, optimizer, tr_data, tr_labels, epoch, True)
        one_epoch(args, model, optimizer, te_data, te_labels, epoch, False)

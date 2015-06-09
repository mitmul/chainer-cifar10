#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', '-f', type=str)
    parser.add_argument('--outfile', '-o', type=str)
    args = parser.parse_args()
    print(args)

    log_fn = args.logfile

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for line in open(log_fn):
        line = line.strip()
        if not line.startswith('epoch'):
            continue
        epoch = int(re.search(ur'epoch:([0-9]+)', line).groups()[0])
        if 'train' in line:
            tr_l = float(re.search(ur'loss=(.+),', line).groups()[0])
            tr_a = float(re.search(ur'accuracy=([0-9\.]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])
            train_acc.append([epoch, tr_a])
        if 'test' in line:
            te_l = float(re.search(ur'loss=(.+),', line).groups()[0])
            te_a = float(re.search(ur'accuracy=([0-9\.]+)', line).groups()[0])
            test_loss.append([epoch, te_l])
            test_acc.append([epoch, te_a])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss[:, 0], train_loss[:, 1], label='training loss')
    ax1.plot(test_loss[:, 0], test_loss[:, 1], label='test loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(train_acc[:, 0], train_acc[:, 1],
             label='training accuracy', c='r')
    ax2.plot(test_acc[:, 0], test_acc[:, 1], label='test accuracy', c='c')
    ax2.set_ylabel('accuracy')

    ax1.legend(loc=0)
    ax2.legend(loc=1)
    plt.savefig(args.outfile)

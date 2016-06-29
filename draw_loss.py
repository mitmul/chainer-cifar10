#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import re
import sys

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def draw_loss_curve(logfile, outfile, epoch=2):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for line in open(logfile):
        line = line.strip()
        if 'epoch:' not in line or 'inf' in line or 'nan' in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        loss = float(re.search('loss:([0-9\.]+)', line).groups()[0])
        acc = float(re.search('accuracy:([0-9\.]+)', line).groups()[0])
        if 'train' in line:
            train_loss.append([epoch, loss])
            train_acc.append([epoch, acc])
        if 'test' in line:
            test_loss.append([epoch, loss])
            test_acc.append([epoch, acc])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)

    if epoch < 2:
        return

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss[:, 0], train_loss[:, 1],
             label='training loss', c='g')
    # ax1.plot(test_loss[:, 0], test_loss[:, 1],
    #  label='test loss', c='g')
    ax1.set_xlim([1, len(train_loss)])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(train_acc[:, 0], train_acc[:, 1],
             label='training accuracy', c='r')
    ax2.plot(test_acc[:, 0], test_acc[:, 1],
             label='test accuracy', c='c')
    ax2.set_xlim([1, len(train_loss)])
    ax2.set_ylabel('accuracy')

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
    plt.savefig(outfile, bbox_inches='tight')

    del fig, ax1, ax2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--outfile', type=str, default='log.png')
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outfile)

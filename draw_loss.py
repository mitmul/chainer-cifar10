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
        if 'train' in line:
            tr_l = float(re.search(ur'loss=(.+),', line).groups()[0])
            tr_a = float(re.search(ur'accuracy=([0-9\.]+)', line).groups()[0])
            train_loss.append(tr_l)
            train_acc.append(tr_a)
        if 'test' in line:
            te_l = float(re.search(ur'loss=(.+),', line).groups()[0])
            te_a = float(re.search(ur'accuracy=([0-9\.]+)', line).groups()[0])
            test_loss.append(te_l)
            test_acc.append(te_a)

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, label='training loss')
    ax1.plot(test_loss, label='test loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.plot(train_acc, label='training accuracy', c='r')
    ax2.plot(test_acc, label='test accuracy', c='c')
    ax2.set_ylabel('accuracy')

    ax1.legend(loc=0)
    ax2.legend(loc=1)
    plt.savefig(args.outfile)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_fn = 'loss.txt'

train_loss = []
train_acc = []
test_loss = []
test_acc = []
for line in open('loss.txt'):
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

plt.plot(train_loss, label='training loss')
plt.plot(train_acc, label='training accuracy')
plt.plot(test_loss, label='test loss')
plt.plot(test_acc, label='test accuracy')
plt.legend()
plt.savefig('loss.jpg')

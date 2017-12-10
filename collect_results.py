#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from collections import OrderedDict
import glob
import json
import os

import numpy as np

import matplotlib.pyplot as plt
from tabulate import tabulate

rows = defaultdict(list)

for dname in glob.glob('results/*'):
    if not os.path.isdir(dname):
        continue
    if not os.path.exists('{}/log'.format(dname)):
        print(dname)
        continue
    log = json.load(open('{}/log'.format(dname)))
    args = json.load(open('{}/args'.format(dname)))
    rows[args['model_name']].append(
        (log[-1]['val/main/accuracy'], log, args, dname))

headers = [
    'model_name',
    'val/main/accuracy',
    'epoch',
    'batchsize',
    'crop_size',
    'expand_ratio',
    'pca_sigma',
    'random_angle',
    'weight_decay',
    'initial_lr',
    'lr_decay_rate',
    'lr_decay_epoch',
]

values = defaultdict(list)
accuracies = {}
for model_name, rows in rows.items():
    rows = sorted(rows, reverse=True)
    for acc, log, args, dname in rows:
        if args['model_name'] in accuracies:
            if acc > accuracies[args['model_name']][-1, 1]:
                accuracies[args['model_name']] = np.array([
                    (l['epoch'], l['val/main/accuracy']) for l in log])
        else:
            accuracies[args['model_name']] = np.array([
                (l['epoch'], l['val/main/accuracy']) for l in log])

        for key, value in log[-1].items():
            if key not in headers:
                continue
            values[key].append(value)
        for key, value in args.items():
            if key not in headers:
                continue
            values[key].append(value)

ordered_values = OrderedDict()
for head in headers:
    ordered_values[head] = values[head]

print(tabulate(ordered_values, headers='keys', tablefmt='pipe'))

for name, accuracy in accuracies.items():
    name = name.split(',')[0]
    plt.plot(accuracy[:, 0], accuracy[:, 1], label=name)
plt.grid()
plt.legend()
plt.savefig('compare.png')

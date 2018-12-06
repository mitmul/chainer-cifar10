#!/bin/bash

MPLBACKEND=Agg python train.py \
--model_file models/wide_resnet.py \
--model_name WideResNet \
--batchsize 128 \
--training_epoch 500 \
--initial_lr 0.05 \
--lr_decay_rate 0.5 \
--lr_decay_epoch 70 \
--weight_decay 0.0005 \
--random_angle 15.0 \
--pca_sigma 25.5 \
--expand_ratio 1.2 \
--crop_size 28 28 \
--seed 0 \
--gpus 0 

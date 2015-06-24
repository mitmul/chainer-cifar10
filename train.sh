#! /bin/bash

python train.py \
--gpu 8 \
--model models/VGG_mini_ABN.py \
--epoch 1000 \
--batchsize 128 \
--prefix VGG_mini_ABN_Adam-0.0001 \
--snapshot 10 \
--datadir data \
--flip 1 \
--shift 10 \
--size 32 \
--norm 0 \
--opt Adam

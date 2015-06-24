#! /bin/bash

python train.py \
--gpu 4 \
--model models/VGG_mini_BN_PReLU.py \
--epoch 1000 \
--batchsize 128 \
--prefix VGG_mini_BN_PReLU_Adam \
--snapshot 10 \
--datadir data \
--flip TRUE \
--shift 5 \
--size 32 \
--norm True \
--opt Adam

#! /bin/bash

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz
rm -rf cifar-10-python.tar.gz
python dataset.py

# -*- coding:utf8 -*-
from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid

feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'convert' ]

feature_num = len(feature_names)

data = numpy.fromfile(filename, sep=' ') # 从文件中读取原始数据
# data.shape = (7084,)

data = data.reshape(data.shape[0] // feature_num, feature_num)
# data.reshape , (506,14)

maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
# data.max(axis=0) 取每列的最大数 axis=0 是按行前进的方向 axis=1 是按列前进的方向
# data.min
# data.sum 行求和，同一列相加
# data.sum(axis=0)/data.shape[0] 求平均数
# array([[ 2,  5,  8,  0],
#      [ 3,  1,  5,  9],
#       [21, 34,  1,  8],
#       [ 3,  7, 11,  0]])
#In[33]: maxa,mina,avgsa=a1.max(axis=0),a1.min(axis=0),a1.sum(axis=0)/a1.shape[0]
#In[34]: print maxa,mina,avgsa
#[21 34 11  9] [2 1 1 0] [ 7 11 6 4]

# import six
for i in six.moves.range(feature_num-1): data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) # six.moves可以兼容python2和python3
# six SIX是用于python2与python3兼容的库。
# data[:, i] 取第i列   data[:i] 是从第一行到第i行
# 归一化 方法：减掉均值，然后除以原取值范围

ratio = 0.8 # 训练集和验证集的划分比例

offset = int(data.shape[0]*ratio)

train_data = data[:offset]

test_data = data[offset:]

train_reader = paddle.batch( paddle.reader.shuffle( train_data, buf_size=500), batch_size=BATCH_SIZE)

test_reader = paddle.batch( paddle.reader.shuffle( test_data, buf_size=500), batch_size=BATCH_SIZ




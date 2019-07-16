#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 12:39 PM
# @Author  : Slade
# @File    : train.py
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
from tools import _readbunchobj, save_path
from data_input import DataInput
# from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
params = _readbunchobj(save_path + "args.param")
train_set = _readbunchobj(save_path + "train_set.dat")

# 参数定义
flags = tf.flags
flags.DEFINE_bool('is_retrain', False, 'if is_retrain is true, not rebuild the summary')
flags.DEFINE_integer('train_data_shape', 10000, 'train data shape[0]')
flags.DEFINE_integer('max_epoch', 1, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 6, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 8e-4, 'initial learning rate, default: 8e-4')
flags.DEFINE_float('decay_rate', 0.75, 'decay rate, default: 0.75')
flags.DEFINE_float('keep_prob', 0.5, 'keep_prob for training, default: 0.5')
flags.DEFINE_integer('batch_size', 50, 'batch_size')

flags.DEFINE_integer('decay_step', 1000, 'decay_step, default: 1000')
flags.DEFINE_integer('valid_step', 500, 'valid_step, default: 500')
flags.DEFINE_float('last_f1', 0.10, 'if valid_f1 > last_f1, save new model. default: 0.10')
FLAGS = flags.FLAGS

lr = FLAGS.lr
last_f1 = FLAGS.last_f1
epoch = FLAGS.max_max_epoch
train_batch_size = FLAGS.batch_size
checkpoint_dir = '/Users/slade/Documents/YMM/Code/tf/model/ckpt'
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    model = Model(args)

    # init variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    time0 = time.time()
    for batch in tqdm(range(epoch)):
        global_step = sess.run(model.global_step)
        for _, uij in DataInput(train_set, train_batch_size):
            # training
            feed_dict = {model.inputs: uij, model.keep_prob: FLAGS.keep_prob, model.lr: lr}
            summary, _cost, _, _ = sess.run(train_fetches, feed_dict)  # the cost is the mean cost of one batch

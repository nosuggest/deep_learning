#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/9 9:59 PM
# @Author  : Slade
# @File    : DataMake.py

import tensorflow as tf
from tensorflow import feature_column as fc

my_feature_columns = []

def truncate(val, left=-1.0, right=1.0):
    return tf.clip_by_value(val, left, right)


def create_feature_columns():
    pass

def parse_example(serial_exmp, feature_spec):
    pass


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size, num_parallel_readers):
    pass


def eval_input_fn(filename, feature_spec):
    pass

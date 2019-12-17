#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 6:54 PM
# @Author  : Slade
# @File    : Iterator.py
import tensorflow as tf


class Iterator:
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _fm_feat_indices, _fm_feat_values, \
        _fm_feat_shape, _labels, _dnn_feat_indices, \
        _dnn_feat_values, _dnn_feat_weights, _dnn_feat_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_values = _fm_feat_values
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weights = _dnn_feat_weights
        self.dnn_feat_shape = _dnn_feat_shape

    def parser(self, record):
        keys_to_features = {
            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_values': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),
            'labels': tf.FixedLenFeature([], tf.string),
            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weights': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], tf.int64), [-1, 2])
        fm_feat_values = tf.sparse_tensor_to_dense(parsed['fm_feat_values'])
        fm_feat_shape = parsed['fm_feat_shape']
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weights = tf.sparse_tensor_to_dense(parsed['dnn_feat_weights'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return fm_feat_indices, fm_feat_values, \
               fm_feat_shape, labels, dnn_feat_indices, \
               dnn_feat_values, dnn_feat_weights, dnn_feat_shape

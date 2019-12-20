#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 4:11 PM
# @Author  : Slade
# @File    : attention.py

import tensorflow as tf


def attention(queries, keys, keys_length):
    '''
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B]
        B: batch_size
        H:  hide_size
        T:  history_length
    '''
    # 是的queries由[B, H]变为[B, T, H]
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H

    # 三层全链接
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1

    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T

    # Mask，去除长短不一的地方为0
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # 在补足的地方附上一个很小的值，而不是0
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T

    # Scale
    # 这个地方和transform中的qk后除以sqrt(dim)的想法一致：用维度的根号来放缩，使得标准化，避免点击过后的值过小：https://github.com/sladesha/Reflection_Summary/blob/master/深度学习/Attention.md#L164
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # B * 1 * T

    # Weighted Sum
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))

    return outputs

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 4:42 PM
# @Author  : Slade
# @File    : dice.py

import tensorflow as tf


def dice(x, training=True, scope=""):
    _x = tf.layers.batch_normalization(x, center=False, scale=False, training=training)
    px = tf.sigmoid(_x)
    alphas = tf.get_variable('alpha' + scope, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return alphas * (1 - px) * _x + px * _x

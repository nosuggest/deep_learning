#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 6:41 PM
# @Author  : Slade
# @File    : extremeDeepFm.py

import numpy as np
import tensorflow as tf
import math
from .Args import HParams
from .Iterator import Iterator


class ExtremeDeepFM():
    '''
    来自：https://github.com/Leavingseason/xDeepFM/tree/master/exdeepfm 的简化版，方便自己理解
    '''

    def __init__(self, src_dataset):
        self.hparams = HParams()
        # xavier_initializer
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.embed_params = []
        self.layer_params = []
        self.cross_params = []
        self.iterator = Iterator(src_dataset)

    def _build_graph(self):
        self.keep_prob_train = 1 - np.array(self.hparams.dropout)
        self.keep_prob_test = np.ones_like(self.hparams.dropout)
        self.layer_keeps = tf.placeholder(tf.float32)
        with tf.variable_scope("exDeepFm") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(name='embedding_layer',
                                                 shape=[self.hparams.FEATURE_COUNT, self.hparams.dim],
                                                 dtype=tf.float32)
                self.embed_params.append(self.embedding)
                embed_out, embed_layer_size = self._build_embedding()
            logit = self._build_linear()
            logit = tf.add(logit, self._build_extreme_FM(embed_out))
            # dnn的input是来自于初始化embedding层的output结果
            logit = tf.add(logit, self._build_dnn(embed_out, embed_layer_size))
            return logit

    def _build_embedding(self):
        '''
        :return: 特征embedding，用作fm，linear，CIN的初始化input
        '''
        fm_sparse_index = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                          self.iterator.dnn_feat_values,
                                          self.iterator.dnn_feat_shape)
        fm_sparse_weight = tf.SparseTensor(self.iterator.dnn_feat_indices,
                                           self.iterator.dnn_feat_weights,
                                           self.iterator.dnn_feat_shape)
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse(self.embedding,
                                                            fm_sparse_index,
                                                            fm_sparse_weight,
                                                            combiner="sum")
        embedding = tf.reshape(w_fm_nn_input_orgin, [-1, self.hparams.dim * self.hparams.FIELD_COUNT])
        embedding_size = self.hparams.FIELD_COUNT * self.hparams.dim
        return embedding, embedding_size

    def _build_linear(self):
        '''
        :return: 线性部分，直接拿feature对应的值进行wx+b即可
        '''
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            w_linear = tf.get_variable(name='w',
                                       shape=[self.hparams.FEATURE_COUNT, 1],
                                       dtype=tf.float32)
            b_linear = tf.get_variable(name='b',
                                       shape=[1],
                                       dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            x = tf.SparseTensor(self.iterator.fm_feat_indices,
                                self.iterator.fm_feat_values,
                                self.iterator.fm_feat_shape)
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w_linear), b_linear)
            self.layer_params.append(w_linear)
            self.layer_params.append(b_linear)
            return linear_output

    def _build_dnn(self, embed_out, embed_layer_size):
        '''
        :param embed_out: onehotencoding后的embedding结果
        :param embed_layer_size: feature个数xdim
        :return:
        '''
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(self.hparams.layer_sizes):
                curr_w_nn_layer = tf.get_variable(name='w_nn_layer' + str(layer_idx),
                                                  shape=[last_layer_size, layer_size],
                                                  dtype=tf.float32)
                curr_b_nn_layer = tf.get_variable(name='b_nn_layer' + str(layer_idx),
                                                  shape=[layer_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx],
                                                       curr_w_nn_layer,
                                                       curr_b_nn_layer)
                scope = "nn_part" + str(idx)

                curr_hidden_nn_layer = self._active_layer(logit=curr_hidden_nn_layer,
                                                     scope=scope,
                                                     layer_idx=idx)
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[last_layer_size, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output

    def _build_extreme_FM(self, nn_input):
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = self.hparams.FIELD_COUNT
        # 特征还原，bacth_size*field_num*dim
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), self.hparams.dim])
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        # 此步骤将bacth_size*field_num*dim切分为dim*[bacth_size,field_num,1]
        split_tensor0 = tf.split(hidden_nn_layers[0], self.hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(self.hparams.cross_layer_sizes):
                split_tensor = tf.split(hidden_nn_layers[-1], self.hparams.dim * [1], 2)
                # 在维度上进行了外积，这就是xdeepfm的CIN的核心
                dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                dot_result_o = tf.reshape(dot_result_m, shape=[self.hparams.dim, -1, field_nums[0] * field_nums[-1]])
                # 恢复成[bacth_size,dim,field_nums[0] * field_nums[-1]]，如果是第一层则为[bacth_size,dim,field_nums[0] * field_nums[0]]
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                filters = tf.get_variable(name="f_" + str(idx),
                                          shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                          dtype=tf.float32)

                # 通过[1,field_nums[-1] * field_nums[0],output]进行卷积处理相当于在dim的每一维上进行融合，融合的是所有特征dim的每一维
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                # Batch * Layer size * dim
                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                if idx != len(self.hparams.cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)

            # sum_pooling操作
            result = tf.concat(final_result, axis=1)
            result = tf.reduce_sum(result, -1)

            # 最后外接一个输出层压缩到1个输出值即可
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[final_len, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exDeepFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

            return exDeepFM_out

    def _active_layer(self, logit, scope, layer_idx):
        logit = tf.nn.dropout(x=logit, keep_prob=self.layer_keeps[layer_idx])
        logit = tf.nn.sigmoid(logit)
        return logit

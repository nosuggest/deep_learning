#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 4:02 PM
# @Author  : Slade
# @File    : model.py

import tensorflow as tf
from .attention import attention
from .dice import dice


class Model:
    def __init__(self, user_count, item_count, cate_count, cate_list):
        '''
        参数含义可以参考youtubenet网络，两者基本一致：http://www.shataowei.com/2018/10/16/YoutubeNet的数据答疑/
        :param user_count:
        :param item_count:
        :param cate_count:
        :param cate_list:
        '''
        # i：当前商品，y：预测结果，hist_i:历史商品点击行为，sl：历史行为长度
        self.i = tf.placeholder(tf.int32, [None, ], name='item')
        self.y = tf.placeholder(tf.float32, [None, ], name='label')
        self.hist_i = tf.placeholder(tf.int32, [None, None], name='history_i')
        self.sl = tf.placeholder(tf.int32, [None, ], name='sequence_length')
        self.lr = tf.placeholder(tf.float64, name='learning_rate')
        hidden_units = 256

        # 商品embedding
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        # 品牌embedding
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        # 获取商品对应的品牌
        self.ic = tf.gather(cate_list, self.i)

        # 拼接商品+品牌的embedding向量
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, self.ic)
        ], axis=1)

        i_b = tf.gather(item_b, self.i)

        # 历史点击行为同样操作
        self.hc = tf.gather(cate_list, self.hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, self.hc),
        ], axis=2)

        hist = attention(i_emb, h_emb, self.sl)
        # attention后在接FC，得到的output就是user_emb
        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, hidden_units])
        # linear
        hist = tf.layers.dense(hist, hidden_units)
        u_emb = hist

        # fcn begin
        din_i = tf.concat([u_emb, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.tanh, name='f1')
        d_layer_1_i = dice(d_layer_1_i, scope='dice_1_i')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.tanh, name='f2')
        d_layer_2_i = dice(d_layer_2_i, scope='dice_2_i')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=tf.tanh, name='f3')

        self.logits = i_b + d_layer_3_i

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # loss and train
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)

        # 以下为predict所需要的all_emb点击概率结果
        u_emb_all = tf.expand_dims(u_emb, 1)
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])

        all_emb = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        all_emb = tf.expand_dims(all_emb, 0)
        all_emb = tf.tile(all_emb, [512, 1, 1])
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.tanh, name='f1', reuse=True)
        d_layer_1_all = dice(d_layer_1_all, scope='dice_1_all')
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.tanh, name='f2', reuse=True)
        d_layer_2_all = dice(d_layer_2_all, scope='dice_2_all')
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=tf.tanh, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])

        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l
        })

        return loss

    def predict(self, sess, i, hist_i, sl):
        return sess.run(self.logits_all, feed_dict={
            self.i: i,
            self.hist_i: hist_i,
            self.sl: sl,
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

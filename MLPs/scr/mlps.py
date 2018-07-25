'''
Created on July 25, 2018

@author: taowei.sha
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
from numpy.random import RandomState


def make_data():
    data = pd.read_csv('/Users/slade/Documents/Personalcode/machine-learning/data/tiny_train_input.csv', header=None,
                       sep=',')
    X = np.array(data.iloc[:, 1:])
    Y = np.array(data.iloc[:, 0])
    return X, Y


class Args():
    is_training = True
    checkpoint_dir = '/Users/slade/Documents/Personalcode/machine-learning/data/saver/ckpt'
    layers_nodes = [256, 1]  # 隐藏层的节点个数，一般在0-1000之间，读者可以自行调整
    learning_rate_base = 1e-5  # 训练weights的速率η
    regularzation_rate = 0.001  # 正则力度
    drop_rate = [1.0, 1.0]


class MLPS(object):
    def __init__(self, args):
        self.is_training = args.is_training
        self.layers_nodes = args.layers_nodes
        self.learning_rate_base = args.learning_rate_base
        self.regularzation_rate = args.regularzation_rate
        self.drop_rate = args.drop_rate
        self.build_model()

    def build_model(self):
        # 设定之后想要被训练的x及对应的正式结果y_
        self.continous_feature = tf.placeholder(tf.float32, [None, 39])
        self.y_ = tf.placeholder(tf.float32, [None, ])

        din_all = self.continous_feature
        # 进行MLPS,同时进行正则
        # 卷积层进行l2正则化，这部分的理论分析可以参考我之前写的：http://www.jianshu.com/p/4f91f0dcba95
        din_all = tf.layers.batch_normalization(inputs=din_all, name='d1')
        self.layer_1 = tf.layers.dense(din_all, self.layers_nodes[0], activation=tf.nn.relu, use_bias=True,
                                       # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularzation_rate),
                                       name='f1')
        # layer_1 = tf.nn.dropout(layer_1, keep_prob=self.drop_rate[0])

        self.layer_2 = tf.layers.dense(self.layer_1, self.layers_nodes[1], activation=tf.nn.sigmoid, use_bias=True,
                                       # kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularzation_rate),
                                       name='f2')
        # layer_2 = tf.nn.dropout(layer_2, keep_prob=self.drop_rate[1])

        if self.is_training:
            self.logits = self.layer_2
            self.global_step = tf.Variable(0, trainable=False)
            # 交叉熵，用来衡量两个分布之间的相似程度
            cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.logits + 1e-24))
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.output)
            # cross_entropy_mean = tf.reduce_mean(cross_entropy)
            self.loss = cross_entropy_mean

            # 我们用learning_rate_base作为基础速率η，来控制梯度下降的速度，对梯度进行限制后计算loss
            opt = tf.train.GradientDescentOptimizer(self.learning_rate_base)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)
        else:
            self.output = self.layer_2

    def train(self, sess, c_data, y_data):
        loss, _, step, logits = sess.run([self.loss, self.train_op, self.global_step, self.logits], feed_dict={
            self.continous_feature: c_data,
            self.y_: y_data
        })
        return loss, step, logits

    def test(self, sess, data):
        result = sess.run([self.output], feed_dict={
            self.continous_feature: data
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


if __name__ == '__main__':
    args = Args()
    X, Y = make_data()
    data = {}
    data['X'] = X
    data['Y'] = Y

    batch_size = 200  # 批量训练的数据，batch_size越小训练时间越长，训练效果越准确（但会存在过拟合）
    length = X.shape[0]
    cnt = length // batch_size
    epoch = 1

    if args.is_training:
        # 根据需要改变dropout的力度，全1.0的过拟合风险比较大
        args.drop_rate = [0.5, 0.5]

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        # build_model
        Model = MLPS(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        if args.is_training:
            # 参数保留
            for k in range(epoch):
                print(k)
                start = 0
                for i in range(1, cnt):
                    end = min(i * batch_size, length)
                    loss, step, logits = Model.train(sess, data['X'][start:end, :], data['Y'][start:end])
                    # print(logits)
                    if i % 10 == 0:
                        print('the times of training is %d, and the loss is %s' % (i, loss))
                        Model.save(sess, args.checkpoint_dir)
                    start = end
        else:
            Model.restore(sess, args.checkpoint_dir)
            result = Model.test(sess, data['X'])
            print(result)
            np.savetxt('../data/result.txt', result[0])
            np.savetxt('../data/Y.txt', data['Y'])

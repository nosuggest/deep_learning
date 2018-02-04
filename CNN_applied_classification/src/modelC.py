from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os
import pickle

# 嵌入矩阵的维度
embed_dim = 32
# cate个数
cate_max = 293
brand_max = 1234
item_max = 73272

# cate_max = max([x[0] for x in features.take(3, 1)]) + 2
# brand_max = max([x[0] for x in features.take(2, 1)]) + 2
# item_max = max([x[0] for x in features.take(1, 1)]) + 2
# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

window_sizes = {5, 10, 15, 20}
filter_num = 8
sentences_size = 50


def get_inputs():
    item = tf.placeholder(tf.int32, [None, 50], name="item")
    brand = tf.placeholder(tf.int32, [None, 30], name="brand")
    cate = tf.placeholder(tf.int32, [None, 15], name="cate")
    targets = tf.placeholder(tf.int32, [None, 2], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return item, brand, cate, targets, LearningRate, dropout_keep_prob


def get_item_categories_layers(cate, brand):
    with tf.name_scope("cate_layers"):
        cate_embed_matrix = tf.Variable(tf.random_uniform([cate_max, embed_dim], -1, 1),
                                        name="cate_embed_matrix")
        cate_embed_layer = tf.nn.embedding_lookup(cate_embed_matrix, cate, name="cate_embed_layer")

        with tf.name_scope("brand_layers"):
            brand_embed_matrix = tf.Variable(tf.random_uniform([brand_max, embed_dim], -1, 1),
                                             name="brand_embed_matrix")
            brand_embed_layer = tf.nn.embedding_lookup(brand_embed_matrix, brand, name="brand_embed_layer")

        if combiner == "sum":
            cate_embed_layer = tf.reduce_sum(cate_embed_layer, axis=1, keep_dims=True)
        brand_embed_layer = tf.reduce_mean(brand_embed_layer, axis=1, keep_dims=True)
        return cate_embed_layer, brand_embed_layer


def get_item_cnn_layer(item, dropout_keep_prob):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        item_embed_matrix = tf.Variable(tf.random_uniform([item_max, embed_dim], -1, 1),
                                        name="movie_title_embed_matrix")
        item_embed_layer = tf.nn.embedding_lookup(item_embed_matrix, item,
                                                  name="movie_title_embed_layer")
        item_embed_layer_expand = tf.expand_dims(item_embed_layer, -1)

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("item_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(item_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_user_feature_layer(cate_embed_layer, brand_embed_layer, dropout_layer):
    with tf.name_scope("bc_fc"):
        # 第一层全连接
        cate_fc_layer = tf.layers.dense(cate_embed_layer, embed_dim, name="cate_fc_layer", activation=tf.nn.relu)
        brand_fc_layer = tf.layers.dense(brand_embed_layer, embed_dim, name="brand_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        bc_combine_layer = tf.concat([cate_fc_layer, brand_fc_layer, dropout_layer], 2)
        bc_combine_layer = tf.contrib.layers.fully_connected(bc_combine_layer, 200, tf.tanh)

        bc_combine_layer_flat = tf.reshape(bc_combine_layer, [-1, 200])
    return bc_combine_layer, bc_combine_layer_flat


# 获取输入占位符
def bulid():
    item, brand, cate, targets, lr, dropout_keep_prob = get_inputs()

    cate_embed_layer, brand_embed_layer = get_item_categories_layers(cate, brand)

    pool_layer_flat, dropout_layer = get_item_cnn_layer(item, dropout_keep_prob)

    item_combine_layer, item_combine_layer_flat = get_user_feature_layer(cate_embed_layer, brand_embed_layer,
                                                                         dropout_layer)
    # 获取电影ID的嵌入向量
    with tf.name_scope("inference"):
        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        inference_layer = item_combine_layer_flat
        inference = tf.layers.dense(inference_layer, 2, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.nn.l2_loss, name="inference")

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        # cost = tf.losses.mean_squared_error(targets, inference)
        cost = tf.losses.softmax_cross_entropy(targets, inference)
        loss = tf.reduce_mean(cost)
        # 优化损失
    #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    return item, brand, cate, targets, lr, dropout_keep_prob, inference, cost, loss, global_step, train_op, item_combine_layer_flat


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


def get_pred_batches(Xs, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end]

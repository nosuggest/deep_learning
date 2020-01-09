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
    # 当我们对类目的分类数未知的时候变量离散化方法，我们通过hash的方式固定指定类目数
    Brand = fc.categorical_column_with_hash_bucket("Brand", 1000)
    # 固定类目下的变量离散化方法
    phoneOs = fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0)
    # 连续变量的处理
    brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0, normalizer_fn=truncate)
    # onehotencoding
    matchType = fc.categorical_column_with_identity("matchType", 9, default_value=0)
    # fc.indicator_column 可以把以上的特征dense化
    return [Brand, phoneOs, brandPrefer, matchType]


def create_embedding_feature_columns(shared_embedding_dim=64):
    '''
    describe:当我们需要对特征进行embedding共享对时候
    :return:
    '''
    # 点击category id
    c1ids = fc.categorical_column_with_hash_bucket("behaviorC1ids", 100, dtype=tf.int64)
    # 对clids进行加权赋值，有点像attention
    c1ids_weighted = fc.weighted_categorical_column(c1ids, "c1idWeights")
    # category id
    c1id = fc.categorical_column_with_hash_bucket("cate1Id", 100, dtype=tf.int64)
    # c1ids_weighted 和 c1id中都用到了category id，但是这边是保证了其在同一个embedding空间，并不是特征一致
    # 此处c1id_emb会返回长度为2的列表，每个元素的是shared_embedding_dim维的tenser，总长2*shared_embedding_dim
    c1id_emb = fc.shared_embedding_columns([c1ids_weighted, c1id], shared_embedding_dim, combiner='sum')
    return c1id_emb


def parse_example(serial_exmp, feature_spec):
    spec = {"click": tf.FixedLenFeature([], tf.int64)}
    spec.update(feature_spec)

    feats = tf.parse_single_example(serial_exmp, features=spec)
    labels = feats.pop('click')
    return feats, labels


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size, num_parallel_readers):
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(lambda x: parse_example(x, feature_spec), num_parallel_calls=8)
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    return dataset


def eval_input_fn(filename, feature_spec, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda x: parse_example(x, feature_spec), num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    return dataset

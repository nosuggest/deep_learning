#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 2:42 PM
# @Author  : Slade
# @File    : TextCNN.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
import shutil
from datetime import date, timedelta

# 参数的统一管理
flags = tf.app.flags
flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate")
flags.DEFINE_float("learning_rate", 0.02, "Learning rate")
flags.DEFINE_integer("embedding_size", 128, "embedding size")
flags.DEFINE_integer("num_filters", 100, "number of filters")
flags.DEFINE_integer("num_classes", 14, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 500000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 55, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "textcnn_data", "Directory containing the dataset")
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for saving model")
flags.DEFINE_string("date_dir", '', "model date partion")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated list of number of window size in each filter")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
flags.DEFINE_string("optimizer", 'Adagrad', "optimizer type {Adam, Adagrad, GD, Momentum}")
flags.DEFINE_string("task_type", 'train', "task type {train, eval, export}")
FLAGS = flags.FLAGS


def clean_text(text):
    '''文本处理'''
    text = re.sub(r"[^A-Za-z0-9\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\`", "\'", text)
    text = text.strip().lower()
    tokens = text.split()
    tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
    return tokens


def clean_data(text, vocab):
    def decode_text(record):
        fields = record.decode().split(",")
        if len(fields) < 3:
            raise ValueError

        # content
        tokens = clean_text(fields[2])
        tokens_length = len(tokens)

        if tokens_length > FLAGS.sentence_max_len:
            '''截断
            TODO:权重/频率采样
            '''
            tokens = tokens[:FLAGS.sentence_max_len]

        if tokens_length < FLAGS.sentence_max_len:
            '''补位'''
            tokens += [FLAGS.pad_word] * (FLAGS.sentence_max_len - tokens_length)
        return [tokens, np.int32(fields[0])]  # fields[0]：label

    result = tf.py_func(decode_text, [text], [tf.string, tf.int32])
    result[0].set_shape([FLAGS.sentence_max_len])
    result[1].set_shape([])
    # 取tokens的seat_id
    ids = vocab.lookup(result[0])
    # - 1是因为label从1开始，平衡到0-3
    return {"sentence": ids}, result[1] - 1


def input_fn(path_csv, path_vocab, shuffle_buffer_size, num_oov_buckets):
    '''
    整体的数据构造都是自动化的，包括shuffle，batch，以及后续的喂数，更加快速便捷
    1.从csv读入数据，三列：label，title，content
    2.num_oov_buckets 为 <pad>
    3.map过程比较重要，关系到没条数据的处理逻辑，clean_data函数是主要要去该的地方
    '''
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
    dataset = tf.data.TextLineDataset(path_csv)
    dataset = dataset.map(lambda text: clean_data(text, vocab))

    # shuffle将数据打乱，数值越大，混乱程度越大
    if shuffle_buffer_size > 0:
        # repeat数据集重复了指定次数
        dataset = dataset.shuffle(shuffle_buffer_size).repeat()

    # 按照顺序取出FLAGS.batch_size行数据，最后一次输出可能小于FLAGS.batch_size
    dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
    return dataset


def my_model(features, labels, mode, params):
    '''
    :param features: input dataset :{"sentence": ids}, result[1] - 1
    :param labels:
    :param mode:
    :param params: {
            'vocab_size': config["vocab_size"],
            'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        }
    :return:
    :description:和之前卷积过程一致：https://github.com/sladesha/deep_learning/blob/master/TextCNN/textcnn_data/text_cnn.py
    '''
    sentence = features['sentence']
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[params["vocab_size"], FLAGS.embedding_size])
    sentence = tf.nn.embedding_lookup(embeddings, sentence)
    sentence = tf.expand_dims(sentence, -1)
    print("sentence size:%s" % (sentence.shape))
    pooled_outputs = []
    for filter_size in params["filter_sizes"]:
        print("conv:%s" % ([filter_size, FLAGS.embedding_size]))
        conv = tf.layers.conv2d(
            sentence,
            filters=FLAGS.num_filters,
            kernel_size=[filter_size, FLAGS.embedding_size],
            strides=(1, 1),
            padding="VALID",
            activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(
            conv,
            pool_size=[FLAGS.sentence_max_len - filter_size + 1, 1],
            strides=(1, 1),
            padding="VALID")
        pooled_outputs.append(pool)
    h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(
        params["filter_sizes"])])  # shape: (batch, len(filter_size) * embedding_size)

    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'],
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.contrib.layers.fully_connected(inputs=h_pool_flat, num_outputs=FLAGS.num_classes, activation_fn=None, \
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg),
                                               scope='deep_out')
    # logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

    predictions = {"prob": logits}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # predict输出
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'], initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(params['learning_rate'])

    # optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

    def _train_op_fn(loss):
        return optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
        return my_head.create_estimator_spec(
            features=features,
            mode=mode,
            labels=labels,
            logits=logits,
            train_op_fn=_train_op_fn
        )


def main(_):
    '''整个textcnn只有两个前置过程：input_fn+my_model,所以看起来比传统的写法更加简洁'''
    '''基础参数：
    {
    "train_size": 820000,
    "test_size": 40000,
    "vocab_size": 420228,
    "pad_word": "<pad>",
    "num_oov_buckets": 1
}
    '''
    # 时间标记ckpt地址
    FLAGS.date_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.date_dir
    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    # 基础参数（词汇量，train/test量）
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    with open(json_path) as f:
        config = json.load(f)

    # 超参数
    params = {
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }
    FLAGS.pad_word = config["pad_word"]
    if config["train_size"] < FLAGS.shuffle_buffer_size:
        FLAGS.shuffle_buffer_size = config["train_size"]
    print("shuffle_buffer_size:", FLAGS.shuffle_buffer_size)

    # 数据加载
    # distinct word
    path_words = os.path.join(FLAGS.data_dir, 'words.txt')
    # train data
    path_train = os.path.join(FLAGS.data_dir, 'train.csv')
    # eval data
    path_eval = os.path.join(FLAGS.data_dir, 'eval.csv')

    # 模型区
    runConfig = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=runConfig
    )
    if FLAGS.task_type == 'train':
        print("start train：")
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
            max_steps=FLAGS.train_steps
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(path_eval, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
            throttle_secs=300)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    elif FLAGS.task_type == 'eval':
        preds = classifier.predict(
            input_fn=lambda: input_fn(path_eval, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
            predict_keys="prob")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))

    elif FLAGS.task_type == 'export':
        feature_spec = {
            'sentence': tf.placeholder(dtype=tf.int64, shape=[None], name='sentence_id')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        classifier.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    # 启动项
    # python TextCNN.py --train_steps=2000 --clear_existing_model=True
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

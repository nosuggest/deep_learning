#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 9:56 AM
# @Author  : Slade
# @File    : Doc2Vec.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import logging

# 参数的统一管理
flags = tf.app.flags
flags.DEFINE_integer("train_steps", 10000, "number of (global) training steps to perform")
flags.DEFINE_integer("save_checkpoints_secs", 720, "save checkpoints per %save_checkpoints_secs seconds")
flags.DEFINE_integer("keep_checkpoint_max", 3, "keep checkpoints amount")
flags.DEFINE_integer("batch_size", 512, "training batch size")
flags.DEFINE_integer("buffer_size", 1024, "dataset read buffer_size")
flags.DEFINE_integer("word_vocab_size", 32000, "the length of word vocabulary")
flags.DEFINE_integer("cate_vocab_size", 1800, "the length of cate vocabulary")
flags.DEFINE_integer("word_embedding_size", 128, "the dim of word embedding vector")
flags.DEFINE_integer("cate_embedding_size", 128, "the dim of cate embedding vector")
flags.DEFINE_string("embedding_merge", 'concat', "embedding vector summary type，optimizer type {concat，avg}")
flags.DEFINE_float("learning_rate", 0.03, "learning rate")
flags.DEFINE_string("optimizer", 'Adagrad', "optimizer type {Adam, Adagrad, GD, Momentum}")
flags.DEFINE_integer("num_negative_samples", 63, "how many negative samples should be used for an instance")
flags.DEFINE_integer("log_step_count_steps", 500, "log_step_count_steps")
flags.DEFINE_integer("tf_random_seed", 0, "random seed")
flags.DEFINE_boolean("evaluate", False, "whether to start evaluation process")
flags.DEFINE_string("checkpointDir", "model_dir", "Directory where checkpoints and event logs are written to.")
flags.DEFINE_string("train_data", "text/data/*.tfrecord", "path to the training data")
flags.DEFINE_string("eval_data", "text/data/*.tfrecord", "path to the evaluation data.")

FLAGS = flags.FLAGS


class Doc2Vec(tf.estimator.Estimator):
    def __init__(self, params, model_dir=None, optimizer=FLAGS.optimizer, config=None):
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8)
        elif optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'],
                                                       initial_accumulator_value=1e-8)
        elif optimizer == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.95)
        elif optimizer == 'ftrl':
            self.optimizer = tf.train.FtrlOptimizer(params['learning_rate'])
        else:
            raise TypeError("not support {0}".format(FLAGS.optimizer))

        def my_model(features, labels, mode, params):
            # 词汇量
            vocabulary_size = params["vocab_size"]
            # 文本量
            doc_vocab_size = params["cate_vocab_size"]
            # embedding长度
            embedding_size = params["embedding_size"]
            doc_embedding_size = params["doc_embedding_size"]

            # Define Embeddings:
            if params["embedding_merge"] == "concat":
                hidden_units = embedding_size + doc_embedding_size
            else:
                if embedding_size == doc_embedding_size:
                    hidden_units = embedding_size
                else:
                    raise TypeError("embedding_size!=doc_embedding_size,not support avg")

            embeddings = tf.get_variable(name="word_embeddings", shape=[vocabulary_size, embedding_size],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-1.0, 1.0))
            doc_embeddings = tf.get_variable(name="doc_embeddings", shape=[doc_vocab_size, doc_embedding_size],
                                             dtype=tf.float32, initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, features["context_word"])
            embed = tf.reduce_sum(embed, -1)
            doc_embed = tf.nn.embedding_lookup(doc_embeddings, features["cate_id"])

            # 去量冈化
            init_stddev = 1.0 / np.sqrt(hidden_units)
            weights = tf.get_variable("weights", [vocabulary_size, hidden_units], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=init_stddev))
            biases = tf.get_variable("biases", [vocabulary_size], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())

            if params["embedding_merge"] == "concat":
                final_embed = tf.concat([embed, doc_embed], axis=1)
            else:
                final_embed = (embed + doc_embed) / 2.0

            # loss计算，采用量基于负采样的损失函数：tf.nn.nce_loss
            num_sampled = params["num_negative_samples"]
            # remove_accidental_hits剔除负采样出来的正样本
            loss = tf.reduce_mean(tf.nn.nce_loss(weights, biases, labels=tf.expand_dims(labels, 1), inputs=final_embed,
                                                 num_sampled=num_sampled, num_classes=vocabulary_size,
                                                 remove_accidental_hits=True))

            global_step = tf.train.get_or_create_global_step()
            train_op = self.optimizer.minimize(loss, global_step=global_step)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(Doc2Vec, self).__init__(model_fn=my_model, model_dir=model_dir, config=config, params=params)


def parser(x, feature_spec):
    features = tf.parse_example(x, feature_spec)
    target = features.pop("target_word")
    return features, target


def input_fn(filepattern, train=True, shuffle_buffer_size=0):
    d = tf.data.Dataset.list_files(filepattern)
    feature_spec = {
        "context_word": tf.FixedLenFeature([], tf.int64, default_value=0),
        "target_word": tf.FixedLenFeature([], tf.int64, default_value=0),
        "cate_id": tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    dataset = d.apply(lambda filename: tf.data.TFRecordDataset(filename, buffer_size=FLAGS.buffer_size))

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if train:
        dataset = dataset.repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(lambda x: parser(x, feature_spec))
    return dataset


def evaluation_module(model, train_files, eval_files):
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_files, True), max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(eval_files, False),
                                      throttle_secs=FLAGS.save_checkpoints_secs, steps=None)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main(_):
    train_files = FLAGS.train_data
    eval_files = FLAGS.eval_data
    model_params = {
        'learning_rate': FLAGS.learning_rate,
        'vocab_size': FLAGS.word_vocab_size,
        'cate_vocab_size': FLAGS.cate_vocab_size,
        'embedding_size': FLAGS.word_embedding_size,
        'doc_embedding_size': FLAGS.cate_embedding_size,
        'num_negative_samples': FLAGS.num_negative_samples,
        'embedding_merge': 'concat'
    }

    if FLAGS.evaluate:
        config = tf.estimator.RunConfig()
        model = Doc2Vec(params=model_params, model_dir=FLAGS.checkpointDir, config=config)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_files, True), max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(eval_files, False),
                                          throttle_secs=FLAGS.save_checkpoints_secs, steps=None)
        return tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    config = tf.estimator.RunConfig(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        log_step_count_steps=FLAGS.log_step_count_steps,
        save_summary_steps=FLAGS.log_step_count_steps)

    if FLAGS.tf_random_seed != 0:
        config.replace(tf_random_seed=FLAGS.tf_random_seed)
    model = Doc2Vec(params=model_params, model_dir=FLAGS.checkpointDir, config=config)
    model.train(lambda: input_fn(train_files, True), max_steps=FLAGS.train_steps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

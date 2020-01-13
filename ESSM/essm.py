#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 10:20 PM
# @Author  : Slade
# @File    : essm.py

import tensorflow as tf
from .utils import *

flags = tf.app.flags
flags.DEFINE_string("model_dir", '', "model check point dir")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate")
flags.DEFINE_float("learning_rate", 0.02, "Learning rate")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
FLAGS = flags.FLAGS

my_feature_columns = []


def input_fn(filenames, batch_size, shuffle_buffer_size, num_epochs=None, ):
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.data.TFRecordDataset)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.map(parse_exmp)
    if num_epochs:
        dataset = dataset.repeat(num_epochs).batch(batch_size).prefetch(1)
    else:
        dataset = dataset.repeat().batch(batch_size).prefetch(1)
    return dataset


def eval_input_fn(filename, batch_size, shuffle_buffer_size=None):
    dataset = tf.data.TFRecordDataset(filename)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.map(parse_exmp)
    dataset = dataset.batch(batch_size)
    return dataset


def my_model():
    pass


def main(_):
    params = {}
    config = tf.estimator.RunConfig().replace(model_dir=FLAGS.model_dir,
                                              save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    essm = tf.estimator.Estimator(model_fn=my_model,
                                  config=config,
                                  params=params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

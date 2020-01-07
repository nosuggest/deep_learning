#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 3:31 PM
# @Author  : Slade
# @File    : DeepFM.py
import shutil
import tensorflow as tf

#################### # 参数的统一管理 ####################
FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_integer("feature_size", 0, "Number of features")
flags.DEFINE_integer("field_size", 0, "Number of fields")
flags.DEFINE_integer("embedding_size", 32, "Embedding size")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("batch_size", 64, "Number of batch size")
flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
flags.DEFINE_float("learning_rate", 0.02, "learning rate")
flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
flags.DEFINE_string("data_dir", '', "data dir")
flags.DEFINE_string("date_dir", '', "model date partion")
flags.DEFINE_string("model_dir", '', "model check point dir")
flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def input_fn():
    pass


def my_model():
    pass


def main(_):
    pass


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

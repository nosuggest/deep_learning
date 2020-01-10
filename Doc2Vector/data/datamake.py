#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 3:23 PM
# @Author  : Slade
# @File    : datamake.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("input_dir", "./data/", "input dir")
flags.DEFINE_string("output_dir", "./text/data/", "output dir")

FLAGS = flags.FLAGS


# 运 两车 西瓜 到 北京 到付
# 23 1023 94 782 4234 10304
def gen_tfrecords(in_file):
    basename = os.path.basename(in_file) + ".tfrecord"
    out_file = os.path.join(FLAGS.output_dir, basename)
    tfrecord_out = tf.python_io.TFRecordWriter(out_file)
    with open(in_file) as fi:
        idx = 0
        for line in fi:
            fields = line.strip().split(' ')
            for i in range(len(fields)):
                content = np.array(fields[max(0, i - 2):i] + fields[i + 1:min(i + 3, len(fields))])
                target = np.array([fields[i]])
                feature = {
                    "context_word": tf.train.Feature(int64_list=tf.train.Int64List(value=content.astype(np.int))),
                    "target_word": tf.train.Feature(int64_list=tf.train.Int64List(value=target.astype(np.int))),
                    "cate_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[idx]))
                }
                # serialized to Example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                tfrecord_out.write(serialized)
    # 数据打包完成
    tfrecord_out.close()


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    gen_tfrecords(FLAGS.input_dir + "test.txt")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

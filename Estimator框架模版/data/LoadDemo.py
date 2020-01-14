#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : LoadDemo.py
@Author: sladesha
@Date  : 2020/1/14 10:33
@Desc  : 
'''

import tensorflow as tf
import jieba
import re

rules = re.compile("[^\u4e00-\u9fa5a-z0-9.]")


def input_fn(filenames="./data/knowledge.txt", batch_size=32, epoch_num=None, shuffle_size=256):
    dataset = tf.data.TextLineDataset(filenames)

    def clean_data(line):
        columns_data = tf.string_split([line], '\t')
        # tensor化
        labels = tf.string_to_number(columns_data.values[1], out_type=tf.float32)
        splits_data = columns_data.values[2]

        def split_word(text):
            text = text.decode()
            print(text)
            text = rules.sub("", text)
            text = text.strip()
            tokens = jieba.lcut(text)
            print(tokens)
            if len(tokens)==0:
                tokens = ["未知空"]
            # dataset需要保证长度一致
            return tokens[:1]
        # tf.py_func 将普通函数作用在tensor上
        result = tf.py_func(split_word, [splits_data], [tf.string])
        return {"context": result}, labels

    dataset = dataset.map(lambda line: clean_data(line))
    # shuffle将数据打乱，数值越大，混乱程度越大
    if shuffle_size > 0:
        if epoch_num:
            # repeat数据集重复了指定次数
            dataset = dataset.shuffle(shuffle_size).repeat(epoch_num)
        else:
            dataset = dataset.shuffle(shuffle_size).repeat()

    # 按照顺序取出FLAGS.batch_size行数据，最后一次输出可能小于FLAGS.batch_size
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

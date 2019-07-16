#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 1:34 PM
# @Author  : Slade
# @File    : build_model_data.py
from tools import _readbunchobj, save_path, _writebunchobj
import random
from tqdm import tqdm


params = _readbunchobj(save_path + "args.param")
random.seed(1234)

trian_data_char = _readbunchobj(save_path + "train_char2id.dat")
trian_data_word = _readbunchobj(save_path + "train_word2id.dat")
trian_data_label = _readbunchobj(save_path + "train_label.dat")

train_set = []
cnt = 0
for index in tqdm(range(params)):
    char_list = trian_data_char[index]
    word_list = trian_data_word[index]

    char_hist = len(char_list)
    word_hist = len(word_list)

    deduplication_char_list = [x for x in char_list if x > 0]
    deduplication_word_list = [x for x in word_list if x > 0]
    deduplication_char_hist = len(deduplication_char_list)
    deduplication_word_hist = len(deduplication_word_list)

    if deduplication_word_hist < 1:
        cnt += 1
        print('Info too short,Total short number is %s' % cnt)
        continue

    train_set.append((char_list, word_list, char_hist, word_hist, deduplication_char_list, deduplication_word_list,
                      deduplication_char_hist, deduplication_word_hist, trian_data_label[index]))

random.shuffle(train_set)
print(train_set[:2])
_writebunchobj(save_path + "train_set.dat", train_set)

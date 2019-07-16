#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 8:01 PM
# @Author  : Slade
# @File    : sample2id.py

from collections import defaultdict
from tools import _writebunchobj, _readbunchobj, rule, jieba
import time
import pandas as pd
import os
import sys
from tqdm import tqdm

path = "/Users/slade/Documents/YMM/Code/tf/basedata/"
save_path = "/Users/slade/Documents/YMM/Code/tf/processeddata/"
withlabel = pd.read_csv(path + "withlabel.csv")


def cut_data(data, desc='result', after='after_cut'):
    data[after] = data[desc].apply(
        lambda x: jieba.lcut(rule.sub("", x.lower())))
    return data


def make_unlabel_set2id():
    unlabel = pd.read_csv(path + "unlabel.csv", header=None)

    print("Making char/word set.")
    time0 = time.time()
    sentence = ''
    for single in tqdm(unlabel.values):
        sentence += single[0]
    char2id_set = set(sentence)
    print('char set number is %d' % len(char2id_set))
    char2id = dict(zip(char2id_set, range(1, len(char2id_set) + 1)))
    char2id.update({"UNK": 0})
    _writebunchobj(save_path + "char2id.dat", char2id)

    unlabel.columns = ['result']
    unlabel = cut_data(data=unlabel)

    sentence = []
    for single in tqdm(unlabel.after_cut.values):
        sentence += single
    word_set = set(sentence)
    print('word set number is %d' % len(word_set))
    word2id = dict(zip(word_set, range(1, len(word_set) + 1)))
    word2id.update({"UNK": 0})
    _writebunchobj(save_path + "word2id.dat", word2id)
    print('Finished get the all unsupervised char/word to ids. Costed time %g s' % (time.time() - time0))


try:
    char2id = _readbunchobj(save_path + "char2id.dat")
    word2id = _readbunchobj(save_path + "word2id.dat")
except:
    make_unlabel_set2id()
    char2id = _readbunchobj(save_path + "char2id.dat")
    word2id = _readbunchobj(save_path + "word2id.dat")


def get_char_id(char):
    return char2id.get(char, char2id.get("UNK"))


def get_chars_id(chars):
    chars = list(chars)
    return [get_char_id(char) for char in chars]


def get_word_id(word):
    return word2id.get(word, word2id.get("UNK"))


def get_words_id(words, if_split=False, sep=','):
    if if_split:
        words = words.strip().split(sep)
    if type(words) != list:
        words = list(words)
    return [get_word_id(word) for word in words]


def dataset_char2id(data=withlabel, get_label=True):
    time0 = time.time()
    print('Processing map data char id.')
    rows = data.shape[0]
    print('train dataset number %d' % rows)
    train_char_id = []
    train_label = []
    for i in tqdm(range(rows)):
        train_char_id.append(get_chars_id(data.iloc[i, 0]))
        if get_label:
            train_label.append(data.iloc[i, 1])
    print('Finished changing the eval chars to ids. Costed time %g s' % (time.time() - time0))
    _writebunchobj(save_path + "train_char2id.dat", train_char_id)
    if get_label:
        _writebunchobj(save_path + "train_label.dat", train_label)
    print("Train data Save finished.")


def dataset_word2id(data):
    data = cut_data(data=data, desc="content")
    time0 = time.time()
    print('Processing map data word id.')
    rows = data.shape[0]
    print('train dataset number %d' % rows)
    train_word_id = []
    for i in tqdm(range(rows)):
        train_word_id.append(get_words_id(data.iloc[i, 2]))
    print('Finished changing the eval word to ids. Costed time %g s' % (time.time() - time0))
    _writebunchobj(save_path + "train_word2id.dat", train_word_id)
    print("Train data Save finished.")


if __name__ == "__main__":
    dataset_char2id(withlabel)
    dataset_word2id(withlabel)

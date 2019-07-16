#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/1 8:44 PM
# @Author  : Slade
# @File    : tools.py

import pickle
import jieba
import re

rule = re.compile("[^\u4e00-\u9fa50-9a-z.]")

path = "/Users/slade/Documents/YMM/Code/tf/basedata/"
save_path = "/Users/slade/Documents/YMM/Code/tf/processeddata/"

jieba.load_userdict(
    "/Users/slade/Documents/YMM/Code/tf/basedata/user_dict.txt")


def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj, 1)


if __name__ == '__main__':
    pass

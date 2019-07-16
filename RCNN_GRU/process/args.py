#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 12:45 PM
# @Author  : Slade
# @File    : args.py
import pandas as pd
from tools import _writebunchobj, save_path


class Args():
    def __init__(self):
        _path = "/Users/slade/Documents/YMM/Code/tf/basedata/"
        _withlabel = pd.read_csv(_path + "withlabel.csv")
        length = _withlabel.shape[0]
        _writebunchobj(save_path + "args.param", (length))


if __name__ == "__main__":
    args = Args()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/17 6:42 PM
# @Author  : Slade
# @File    : Args.py

class HParams:
    def __init__(self):
        self.dropout = 0.5
        # onehotencoding之前的特征维度
        self.FIELD_COUNT = 55
        # onehotencoding之后的特征维度
        self.FEATURE_COUNT = 210
        self.dim = 256
        self.layer_sizes = [512, 256, 128]
        self.cross_layer_sizes = [256, 256, 256]

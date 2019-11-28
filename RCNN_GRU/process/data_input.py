#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 3:43 PM
# @Author  : Slade
# @File    : data_input.py
import numpy as np
import random
import pysnooper

random.seed(234)


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    # @pysnooper.snoop
    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        print(ts)
        self.i += 1
        cl_ret, wl_ret, y = [], [], []
        for t in ts:
            print(t)
            if t[2] <= 50:
                cl = t[0] + [0] * (50 - t[2])
            else:
                # 开头结尾是重灾区
                start = random.sample(range(int(t[2] * 0.2)), 1)[0]
                end = random.sample(range(int((t[2] - 26) * 0.8), t[2] - 26), 1)[0]
                cl = t[0][start:(start + 25)] + t[0][end:(end + 25)]

            if t[3] <= 24:
                wl = t[1] + [0] * (24 - t[3])
            else:
                start = random.sample(range(int(t[3] * 0.2)), 1)[0]
                end = random.sample(range(int((t[3] - 13) * 0.8), t[3] - 13), 1)[0]
                wl = t[1][start: (start + 12)] + t[1][end:(end + 12)]

            if t[6] <= 25:
                dcl = t[4] + [0] * (25 - t[6])
            else:
                idx = random.randint(0, t[6] - 26)
                dcl = t[4][idx:(idx + 25)]

            if t[7] <= 12:
                dwl = t[5] + [0] * (12 - t[7])
            else:
                idx = random.randint(0, t[7] - 13)
                dwl = t[5][idx:(idx + 12)]
            cl_ret.append(cl + dcl)
            wl_ret.append(wl + dwl)

            y.append(t[8])
        return self.i, (cl_ret, wl_ret, y)

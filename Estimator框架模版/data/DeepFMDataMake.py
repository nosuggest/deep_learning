#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/9 10:05 AM
# @Author  : Slade
# @File    : DeepFMDataMake.py
import sys
import random
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=2, help="threads num")
parser.add_argument("--input_dir", type=str, default="", help="input data dir")
parser.add_argument("--output_dir", type=str, default="", help="feature map outputs dir")
parser.add_argument("--mincutoff", type=int, default=200, help="cutoff long-tailed categorical values")
FLAGS, unparsed = parser.parse_known_args()

continous_features = range(1, 14)
continous_clip = [988, 497, 638, 78, 790, 575, 120, 822, 852, 74, 625, 630, 247]
categorial_features = range(14, 40)


class CategoryDictGenerator:
    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            # dict存储每一维特征的细节
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, mincutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        # 统计每一维度上度不同类别度样本量
                        self.dicts[i][features[categorial_features[i]]] += 1

        for i in range(0, self.num_feature):
            # 清洗低频的类别
            self.dicts[i] = filter(lambda x: x[1] >= mincutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))

            # 重新编码
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))


class ContinuousFeatureGenerator:
    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        # 异常点处理
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        # 最小值更新
                        self.min[i] = min(self.min[i], val)
                        # 最大值更新
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        # 标准化
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


def preprocess(datadir, outdir):
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(FLAGS.input_dir + 'train.txt', continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(FLAGS.input_dir + 'train.txt', categorial_features, mincutoff=FLAGS.mincutoff)

    with open(FLAGS.output_dir + 'feature_map', 'w') as output:
        for i in continous_features:
            output.write("{0} {1}\n".format('I' + str(i), i))

        dict_sizes = dicts.dicts_sizes()
        categorial_feature_offset = [dists.num_feature]

        for i in range(1, len(categorial_features) + 1):
            offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
            categorial_feature_offset.append(offset)
            for key, val in dicts.dicts[i - 1].items():
                output.write("{0} {1}\n".format('C' + str(i) + '|' + key, categorial_feature_offset[i - 1] + val + 1))

    random.seed(0)

    with open(FLAGS.output_dir + 'tr.dat', 'w') as out_train:
        with open(FLAGS.output_dir + 'va.dat', 'w') as out_valid:
            with open(FLAGS.input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        feat_vals.append(
                            str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]

                    # train/test切分
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(FLAGS.output_dir + 'te.dat', 'w') as out:
        with open(FLAGS.input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))


if __name__ == "__main__":
    preprocess(FLAGS.input_dir, FLAGS.output_dir)

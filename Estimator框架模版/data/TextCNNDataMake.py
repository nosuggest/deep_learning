#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 12:01 PM
# @Author  : Slade
# @File    : TextCNNDataMake.py

import argparse
from collections import Counter
import json
import os
import re
'''
文本处理不类似于ctr处理，不会借助大量的from tensorflow import feature_column的处理方式
'''
parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum words count", type=int)
parser.add_argument('--data_dir', default='./Estimator框架模版/model/textcnn_data', help="Containing dataset directory",
                    type=str)
parser.add_argument('--num_oov_buckets', default=1, help="Num oov buckets", type=int)

# Hyper parameters for the vocab
PAD_WORD = None


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\`", "\'", text)
    text = text.strip().lower()
    return text


def update_vocab(txt_path, vocab):
    with open(txt_path) as f:
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            if len(fields) < 3: continue
            text = clean_str(fields[2])
            tokens = text.split()
            tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
            vocab.update(tokens)
    return i + 1


if __name__ == '__main__':
    args = parser.parse_args()
    NUM_OOV_BUCKETS = args.num_oov_buckets
    PAD_WORD = '<pad>'

    '''
    train.csv/test.csv数据格式：
    "1"\t"NSPCL"\t"NSPCL (NTPC-SAIL Power Company Limited) is a joint venture company of NTPC Limited and SAIL to generate power for various steel plants throughout India."\n
    '''
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train.csv'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test.csv'), words)

    words = [tok for tok, count in words.items() if count >= args.min_count_word]

    if PAD_WORD not in words: words.append(PAD_WORD)

    with open(os.path.join(args.data_dir, 'words.txt'), "w") as f:
        f.write("\n".join(token for token in words))

    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,
        'pad_word': PAD_WORD,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }

    with open(os.path.join(args.data_dir, 'dataset_params.json'), 'w') as f:
        d = {k: v for k, v in sizes.items()}
        json.dump(d, f, indent=4)

import pickle
import pandas as pd
import numpy as np


def load_data():
    train_data = {}

    file_path = '../data/tiny_train_input.csv'
    data = pd.read_csv(file_path, header=None)
    data.columns = ['c' + str(i) for i in range(data.shape[1])]
    label = data.c0.values
    label = label.reshape(len(label), 1)
    train_data['y_train'] = label

    co_feature = pd.DataFrame()
    ca_feature = pd.DataFrame()
    ca_col = []
    co_col = []
    feat_dict = {}
    cnt = 1
    for i in range(1, data.shape[1]):
        target = data.iloc[:, i]
        col = target.name
        l = len(set(target))
        if l > 10:
            target = (target - target.mean()) / target.std()
            co_feature = pd.concat([co_feature, target], axis=1)
            feat_dict[col] = cnt
            cnt += 1
            co_col.append(col)
        else:
            us = target.unique()
            print(us)
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))
            ca_feature = pd.concat([ca_feature, target], axis=1)
            cnt += len(us)
            ca_col.append(col)
    feat_dim = cnt
    feature_value = pd.concat([co_feature, ca_feature], axis=1)
    feature_index = feature_value.copy()
    for i in feature_index.columns:
        if i in co_col:
            feature_index[i] = feat_dict[i]
        else:
            feature_index[i] = feature_index[i].map(feat_dict[i])
            feature_value[i] = 1.

    train_data['xi'] = feature_index.values.tolist()
    train_data['xv'] = feature_value.values.tolist()
    train_data['feat_dim'] = feat_dim
    return train_data

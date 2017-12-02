# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mt
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import sys

# 数据合并
dealed_data = dealed_data['top10'] + [" "] + dealed_data['top9'] + [" "] + dealed_data['top8'] + [" "] + \
              dealed_data['top7'] + [" "] + dealed_data['top6'] + [" "] + dealed_data['top5'] + [" "] + dealed_data[
                  'top4'] + [" "] + dealed_data['top3'] + [" "] + dealed_data['top2'] + [" "] + dealed_data['top1']

# 数据分列
dealed_data = [s.encode('utf-8').split() for s in dealed_data]

# 数据拆分
train_data, test_data = train_test_split(dealed_data, test_size=0.3, random_state=42)

# 原始数据训练
# sg=1,skipgram;sg=0,SBOW
# hs=1:hierarchical softmax,huffmantree
# nagative = 0 非负采样
model = word2vec.Word2Vec(train_data, sg=1, min_count=10, window=2, hs=1, negative=0)

# 最后一次购物最相似的商品组top3
x = 1000
result = []
result = pd.DataFrame(result)
for i in range(x):
    test_data_split = [s.encode('utf-8').split() for s in test_data[i]]
    k = len(test_data_split)
    last_one = test_data_split[k - 1]
    last_one_recommended = model.most_similar(last_one, topn=3)
    tmp = last_one_recommended[0] + last_one_recommended[1] + last_one_recommended[2]
    last_one_recommended = pd.concat([pd.DataFrame(last_one), pd.DataFrame(np.array(tmp))], axis=0)
    last_one_recommended = last_one_recommended.T
    result = pd.concat([pd.DataFrame(last_one_recommended), result], axis=0)

# 倒数第二次购物最相似的商品组top3
x = 1000
result1 = []
result1 = pd.DataFrame(result1)
for i in range(x):
    test_data_split = [s.encode('utf-8').split() for s in test_data[i]]
    k = len(test_data_split)
    last_one = test_data_split[k - 2]
    last_one_recommended = model.most_similar(last_one, topn=3)
    tmp = last_one_recommended[0] + last_one_recommended[1] + last_one_recommended[2]
    last_one_recommended = pd.concat([pd.DataFrame(last_one), pd.DataFrame(np.array(tmp))], axis=0)
    last_one_recommended = last_one_recommended.T
    result1 = pd.concat([pd.DataFrame(last_one_recommended), result1], axis=0)

# 向量库
rbind_data = pd.concat(
    [order_data['top1'], order_data['top2'], order_data['top3'], order_data['top4'], order_data['top5'],
     order_data['top6'], order_data['top7'], order_data['top8'], order_data['top9'], order_data['top10']], axis=0)


# rbind_data_level = pd.Series(rbind_data).unique()
# number = len(rbind_data_level)


def item_probility(model, items, backupitems):
    # model 训练模型
    # item  已知商品组单条商品
    # backupitems 向量库
    base_score = np.array(-10000000000000000000000)
    name = []

    rbind_data = backupitems
    rbind_data_level = pd.Series(rbind_data).unique()
    number = len(rbind_data_level)

    model = model

    tmp = items
    k = len(tmp)
    last_one = tmp[k - 2]
    tmp1 = tmp[0:(k - 1)]
    for i in range(number):
        tmp2 = tmp1[:]
        tmp3 = rbind_data_level[i]
        tmp2.append(tmp3)
        tmp4 = [tmp2]
        newscore = model.score(tmp4)
        if newscore >= base_score:
            base_score = newscore
            name = tmp3
        else:
            pass
    return base_score,last_one,name


# 判断可能性
if __name__ == '__main__':
  order_data = sys.agrv[1]
  dealed_data = order_data.drop('member_id', axis=1)
  dealed_data = pd.DataFrame(dealed_data).fillna(value='')
  x = 50
  start = []
  output = []
  score_final = []
  for i in range(x):
      score = np.array(-100000000000000)
      name = np.array(-100000000000000)
      newscore = np.array(-100000000000000)
      tmp = test_data[i]
      k = len(tmp)
      last_one = tmp[k - 2]
      tmp = tmp[0:(k - 1)]
      for j in range(number):
          tmp1 = tmp[:]
          target = rbind_data_level[j]
          tmp1.append(target)
          test_data_split = [tmp1]
          newscore = model.score(test_data_split)
          if newscore > score:
              score = newscore
              name = tmp1[len(tmp1) - 1]
          else:
              pass
      start.append(last_one)
      output.append(name)
      score_final.append(score)

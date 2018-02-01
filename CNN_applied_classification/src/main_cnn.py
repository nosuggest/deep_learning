import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from .datapreprocessC import load_data
import tensorflow as tf
import os
import pickle
from .modelC import bulid, get_batches

# 加载数据，first rime
features, targets_values = load_data()
pickle.dump((features, targets_values), open('preprocessmine50.p', 'wb'))

# 加载数据，later
features, targets_values = pickle.load(open('preprocessmine50.p', mode='rb'))

targets_values = pd.DataFrame(targets_values)
targets_values.columns = ['title']
targetsmap = {0: [0, 1], 1: [1, 0]}
targets_values = np.array(targets_values['title'].map(targetsmap))
targets_values = np.array([x for x in targets_values])

# 嵌入矩阵的维度
# embed_dim = 32
# cate个数
# cate_max = max([x[0] for x in features.take(3, 1)]) + 2
# brand_max = max([x[0] for x in features.take(2, 1)]) + 2
# item_max = max([x[0] for x in features.take(1, 1)]) + 2
# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
# combiner = "sum"
#
# window_sizes = {5, 10, 15, 20}
# filter_num = 8
# sentences_size = 50

# # Number of Epochs
num_epochs = 5
# # Batch Size
batch_size = 256
dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './run/save'

tf.reset_default_graph()
train_graph = tf.Graph()

item, brand, cate, targets, lr, dropout_keep_prob, inference, cost, loss, global_step, train_op, item_combine_layer_flat = bulid()

losses = {'train': [], 'test': []}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):

        # 将数据集分成训练集和测试集，随机种子不固定
        train_X, test_X, train_y, test_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.2,
                                                            random_state=0)

        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)

        # 训练的迭代，保存训练损失
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            cate_id = np.zeros([batch_size, 15], dtype='int32')
            for i in range(batch_size):
                cate_id[i] = x.take(3, 1)[i]

            item_id = np.zeros([batch_size, 50], dtype='int32')
            for i in range(batch_size):
                item_id[i] = x.take(1, 1)[i]

            brand_id = np.zeros([batch_size, 30], dtype='int32')
            for i in range(batch_size):
                brand_id[i] = x.take(2, 1)[i]

            feed = {
                item: item_id,
                brand: brand_id,
                cate: cate_id,
                targets: np.reshape(y, [batch_size, 2]),
                dropout_keep_prob: dropout_keep,
                lr: learning_rate
            }

            step, train_loss, _ = sess.run([global_step, loss, train_op], feed)  # cost
            losses['train'].append(train_loss)

            # Show every <show_every_n_batches> batches
            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

        # 使用测试数据的迭代
        for batch_i in range(len(test_X) // batch_size):
            x, y = next(test_batches)

            cate_id = np.zeros([batch_size, 15], dtype='int32')
            for i in range(batch_size):
                cate_id[i] = x.take(3, 1)[i]

            item_id = np.zeros([batch_size, 50], dtype='int32')
            for i in range(batch_size):
                item_id[i] = x.take(1, 1)[i]

            brand_id = np.zeros([batch_size, 30], dtype='int32')
            for i in range(batch_size):
                brand_id[i] = x.take(2, 1)[i]

            feed = {
                item: item_id,
                brand: brand_id,
                cate: cate_id,
                targets: np.reshape(y, [batch_size, 2]),
                dropout_keep_prob: 1,
                lr: learning_rate}

            step, test_loss = sess.run([global_step, loss], feed)  # cost

            # 保存测试损失
            losses['test'].append(test_loss)

            if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model
    saver.save(sess, save_dir)  # , global_step=epoch_i
    print('Model Trained and Saved')
print('save successfully!')

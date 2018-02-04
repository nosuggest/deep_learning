import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from .modelC import bulid, get_pred_batches

features, _ = pickle.load(open('preprocessmine50.p', mode='rb'))

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

item, brand, cate, targets, lr, dropout_keep_prob, inference, cost, loss, global_step, train_op, item_combine_layer_flat = bulid()

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "./run/save")

sess.run(tf.global_variables_initializer())

predict_batches = get_pred_batches(features, batch_size)

user_result = []
item_result = []
for batch_i in range(len(features) // batch_size):
    x = next(predict_batches)

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
        dropout_keep_prob: 1
    }

    item_state = np.array(sess.run([item_combine_layer_flat], feed)).reshape(-1, 200)
    item_result.append(item_state)
    user_result.append(x.take(0, 1))
    print('we have reach %s times' % batch_i)
userdata = np.stack(user_result).reshape(1, -1)[0]
itemdata = np.stack(item_result).reshape(-1, 200)

pickle.dump((userdata, itemdata), open('outputvector.p', 'wb'))

print('save successfully!')

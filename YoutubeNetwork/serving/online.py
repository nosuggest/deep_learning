import os
import time
import pickle
import sys
import random
from input import DataInput, DataInputTest
import numpy as np
import tensorflow as tf
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

__author__ = 'sladesal'

class Args():
    is_training = False
    embedding_size = 256
    brand_list = None
    msort_list = None
    item_count = -1
    brand_count = -1
    msort_count = -1

# load data
with open('/Data/sladesha/youtube/Neg_Data/dataset_mine_add.pkl', 'rb') as f:
    train_set_1 = pickle.load(f)
    train_set_2 = pickle.load(f)
    train_set_3 = pickle.load(f)
    test_set = pickle.load(f)
    brand_list = pickle.load(f)
    msort_list = pickle.load(f)
    user_count, item_count, brand_count, msort_count = pickle.load(f)
    item_key, brand_key, msort_key, user_key = pickle.load(f)
print('user_count: %d\titem_count: %d\tbrand_count: %d\tmsort_count: %d' %
      (user_count, item_count, brand_count, msort_count))

# set values
train_set = train_set_1 + train_set_2 + train_set_3
print('train set size', len(train_set))

# init args
args = Args()
args.brand_list = brand_list
args.msort_list = msort_list
args.item_count = item_count
args.brand_count = brand_count
args.msort_count = msort_count

# else para
epoch = 6
train_batch_size = 32
test_batch_size = 50
checkpoint_dir = 'save_path/ckpt'
save_version = '1'


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
model = Model(args)
sess.run(tf.global_variables_initializer())    
sess.run(tf.local_variables_initializer())
model.restore(sess, checkpoint_dir)    

# save exporter data format
saver = tf.train.Saver() 
model_exporter = tf.contrib.session_bundle.exporter.Exporter(saver)
model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
        'hist_click': tf.contrib.session_bundle.exporter.generic_signature({'hist_click': model.hist_click}),
        'sl': tf.contrib.session_bundle.exporter.generic_signature({'sl': model.sl}),
        'last_click': tf.contrib.session_bundle.exporter.generic_signature({'last_click': model.last_click}),
        'basic': tf.contrib.session_bundle.exporter.generic_signature({'basic': model.basic}),
        'keep_prob': tf.contrib.session_bundle.exporter.generic_signature({'keep_prob': model.keep_prob}),
        'outputs': tf.contrib.session_bundle.exporter.generic_signature({'y': model.output})})
model_exporter.export('/Data/sladesha/youtube/offline/test/',tf.constant(save_version), sess)

# start online serve
# bazel build //tensorflow_serving/model_servers:tensorflow_model_server
# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9005 --model_name=test --model_base_path=/Data/sladesha/youtube/offline/test/
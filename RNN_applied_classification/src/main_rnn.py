# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import pred
# import evaluation,pred

PATH_TO_TRAIN = 'Traindata.csv'
PATH_TO_TEST = 'Testdata.csv'

class Args():
    is_training = True
    layers = 1
    rnn_size = 20
    n_epochs = 1
    batch_size = 50
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'uid'
    risk_id = 'item_id'
    time_key = 'time_id'
    grad_cap = 0
    test_model = 0
    checkpoint_dir = 'model'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_risks = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Risk args')
    parser.add_argument('--layer', default=5, type=int)
    parser.add_argument('--rnn_size', default=100, type=int)
    parser.add_argument('--n_epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--dropout_p_hidden', default=0.5, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--pred', default=0, type=int)
    parser.add_argument('--test', default=0, type=int)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--outfile', default='pred', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN)
    valid = pd.read_csv(PATH_TO_TEST)
    args = Args()
    args.n_risks = len(data['item_id'].unique())
    print ('args.n_items:',args.n_risks)
    args.layers = command_line.layer
    args.rnn_size = command_line.rnn_size
    args.n_epochs = command_line.n_epochs
    args.learning_rate = command_line.learning_rate
    args.is_training = command_line.train
    args.is_pred = command_line.pred
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.outfile = command_line.outfile 
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout_p_hidden
    print(args.dropout_p_hidden)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    #gpu_config = tf.ConfigProto(per_process_gpu_memory_fraction=0.1)
    gpu_config = tf.ConfigProto()

    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Risk(sess, args)
        if args.is_training:
            gru.fit(data)
        elif args.is_pred:
            pred.pred_sessions_batch(gru,data,valid,args.outfile)
        # else:
        #     res = evaluation.evaluate_sessions_batch(gru, data, valid)
        #     print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 10:17 AM
# @Author  : Slade
# @File    : model.py

import json
import pickle as cPickle
import tensorflow as tf
from bert_base.bert import modeling, optimization, tokenization
from bert_base.bert.optimization import create_optimizer
from bert_base.train import tf_metrics
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers


class Model():
    def __init__(self):
        self.nums_tags = 4
        self.embedding_size = 50
        self.max_epoch = 10
        self.learning_rate = 0.5
        self.lstm_dim = 128
        self.global_steps = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.checkpoint_dir = "./model/"
        self.checkpoint_path = "./model/train_model.ckpt"
        self.initializer = initializers.xavier_initializer()
        self.entry = "train"
        self.vocab_dir = None
        self.init_checkpoint = None
        self.bert_config = None
        self.is_training = True if self.entry == "train" else False

    def __creat_model(self):

        # embbeding layer
        self._init_bert_placeholder()
        self.bert_layer()
        # bi-Lstm layer
        self.biLSTM_layer()
        # logits_layer
        self.logits_layer()
        # crf loss_layer
        self.loss_layer_crf()

        # classify loss_layer
        # self.loss_layer()

        # optimizer_layer
        self.bert_optimizer_layer()

    def _init_bert_placeholder(self):
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_segment_ids"
        )
        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_targets"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="bert_dropout"
        )
        # 计算补位之前的文本长度
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        self.nums_steps = tf.shape(self.input_ids)[-1]

    def bert_layer(self):
        bert_config = modeling.BertConfig.from_json_file(self.bert_config)

        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # 获取各词embedding结果
        self.embedded_sentence = model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(
            self.embedded_sentence, self.dropout
        )

        # 获取整句embedding结果
        self.embedded_pooled = model.get_pooled_output()
        self.model_inputs_1 = tf.nn.dropout(
            self.embedded_pooled, self.dropout
        )

    def biLSTM_layer(self):
        with tf.variable_scope("bi-LSTM") as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.GRUCell(
                        num_units=self.lstm_dim,
                        # use_peepholes=True,
                        # initializer=self.initializer,
                        # state_is_tuple=True
                    )

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell['forward'],
                cell_bw=lstm_cell['backward'],
                inputs=self.model_inputs,
                sequence_length=self.length,
                dtype=tf.float32,
            )
            self.lstm_outputs = tf.concat(outputs, axis=2)

    def logits_layer(self):
        with tf.variable_scope("hidden"):
            w = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim],
                                dtype=tf.float32, initializer=self.initializer
                                )
            b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                initializer=self.initializer
                                )

            output = tf.reshape(self.lstm_outputs, shape=[-1, self.lstm_dim * 2])
            hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
            self.hidden = hidden

        with tf.variable_scope("logits"):
            w = tf.get_variable("W", shape=[self.lstm_dim, self.nums_tags],
                                initializer=self.initializer, dtype=tf.float32
                                )
            self.test_w = w
            b = tf.get_variable("b", shape=[self.nums_tags], dtype=tf.float32)
            self.test_b = b
            pred = tf.nn.xw_plus_b(hidden, w, b)
            self.logits = tf.reshape(
                pred, shape=[-1, self.nums_steps, self.nums_tags])

    def loss_layer_crf(self):
        with tf.variable_scope("loss_layer"):
            logits = self.logits
            targets = self.targets

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.nums_tags, self.nums_tags],
                initializer=self.initializer
            )

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.length
            )
            self.loss = tf.reduce_mean(-log_likelihood)

    def loss_layer(self):
        with tf.variable_scope("loss_layer"):
            logits = self.logits
            targets = self.targets
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            per_example_loss = -tf.reduce_sum(targets * log_probs, axis=-1)
            self.loss = tf.reduce_mean(-per_example_loss)

    def bert_optimizer_layer(self):
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 2), tf.cast(self.targets, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_train_steps = int(
            self.train_length / self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.train_op = create_optimizer(
            self.loss, self.learning_rate, num_train_steps, num_warmup_steps, False
        )
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def bert_step(self, sess, batch):
        ntokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)

        feed = {
            self.input_ids: inputs_ids,
            self.targets: tag_ids,
            self.segment_ids: segment_ids,
            self.input_mask: input_mask,
            self.dropout: 0.5
        }
        embedding, global_steps, loss, _, logits, acc, length = sess.run(
            [self.embedded_sentence, self.global_steps, self.loss, self.train_op, self.logits, self.accuracy,
             self.length],
            feed_dict=feed)
        return global_steps, loss, logits, acc, length

    def train(self):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_dir,
        )
        self.train_data = None
        self.dev_data = None
        self.dev_batch = None

        data = {
            "batch_size": self.train_data.batch_size,
            "input_size": self.train_data.input_size,
            "vocab": self.train_data.vocab,
            "tag_map": self.train_data.tag_map,
        }

        self.batch_size = self.train_data.batch_size
        self.nums_tags = len(self.train_data.tag_map.keys())
        self.tag_map = self.train_data.tag_map
        self.train_length = len(self.train_data.data)

        # self.test_data = DataBatch(data_type='test', batch_size=100)
        # self.test_batch = self.test_data.get_batch()
        # save vocab


        self.__creat_model()
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                tvars = tf.trainable_variables()
                (assignment_map, initialized_variable_names) = \
                    modeling.get_assignment_map_from_checkpoint(tvars,
                                                                self.init_checkpoint)
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                for i in range(self.max_epoch):
                    steps = 0
                    for batch in self.train_data.get_batch():
                        steps += 1
                        global_steps, loss, logits, acc, length = self.bert_step(
                            sess, batch
                        )
                        if steps % 1 == 0:
                            print("[->] step {}/{}\tloss {:.2f}\tacc {:.2f}".format(
                                steps, len(self.train_data.batch_data), loss, acc))
                    self.saver.save(sess, self.checkpoint_path)

    def decode(self, scores, lengths, trans):
        paths = []
        for score, length in zip(scores, lengths):
            path, _ = viterbi_decode(score, trans)
            paths.append(path)
        return paths

    def prepare_bert_pred_data(self, text):
        tokens = list(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        feed = {
            self.input_ids: [input_ids],
            self.segment_ids: [segment_ids],
            self.input_mask: [input_mask],
            self.dropout: 1
        }
        return feed

    def predict(self, text="今年又变垃圾了呢"):
        self.batch_size = 1
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_dir,
        )
        self.train_length = 10

        self.tag_map = None
        self.nums_tags = len(self.tag_map.values())
        self.__creat_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            trans = self.trans.eval()

            feed = self.prepare_bert_pred_data(text)
            logits, length = sess.run(
                [self.logits, self.length], feed_dict=feed)
            paths = self.decode(logits, length, trans)
            return paths

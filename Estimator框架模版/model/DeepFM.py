#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 3:31 PM
# @Author  : Slade
# @File    : DeepFM.py
import shutil
import tensorflow as tf
from datetime import date, timedelta
import os

# 参数的统一管理
FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_integer("feature_size", 0, "Number of features")
flags.DEFINE_integer("field_size", 0, "Number of fields")
flags.DEFINE_integer("embedding_size", 32, "Embedding size")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("batch_size", 64, "Number of batch size")
flags.DEFINE_float("learning_rate", 0.02, "learning rate")
flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("data_dir", '', "data dir")
flags.DEFINE_string("date_dir", '', "model date partion")
flags.DEFINE_string("model_dir", '', "model check point dir")
flags.DEFINE_string("task_type", 'train', "task type {train, test, eval, export}")
flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def input_fn(filenames, batch_size=64, num_epochs=None, shuffle_buffer_size=None):
    def decode_data(line):
        # line = "1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1"
        # 切分数据
        columns_data = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns_data.values[0], out_type=tf.float32)
        splits_data = tf.string_split(columns_data.values[1:], ':')
        id_vals = tf.reshape(splits_data.values, splits_data.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    dataset = tf.data.TextLineDataset(filenames).map(decode_data, num_parallel_calls=10).prefetch(
        1000)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # None为不指定Epoch轮数
    dataset = dataset.repeat(num_epochs)
    # batch_size
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def my_model(features, labels, mode, params):
    '''
    :param features: 特征数据
    :param labels: 标签数据
    :param mode: 模式（训练，评估，预测..)
    :param params: 超参数
    :return:
    '''
    # feature个数
    field_size = params["field_size"]
    # feature one-hot后的维度
    feature_size = params["feature_size"]
    # embedding维度
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]

    # [256,128,64]
    layers = list(map(int, params["deep_layers"].split(',')))
    # [0.5,0.5,0.5]
    dropout = list(map(float, params["dropout"].split(',')))

    # 参数
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
                           initializer=tf.glorot_normal_initializer())

    # feature处理
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # 这边是基于field去算
    # 一次项
    with tf.variable_scope("first_order"):
        feat_w = tf.nn.embedding_lookup(FM_W, feat_ids)
        y_w = tf.reduce_sum(tf.multiply(feat_w, feat_vals), 1)
    # 二次项
    with tf.variable_scope("second_order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)

    with tf.variable_scope("deep_part"):
        is_train = False
        if FLAGS.batch_norm and mode == tf.estimator.ModeKeys.TRAIN:
            is_train = True

        def BN(x, train, scope):
            bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                                    updates_collections=None, is_training=True, reuse=None,
                                                    scope=scope)
            # 预测的时候，is_training=false
            bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                                    updates_collections=None, is_training=False, reuse=True,
                                                    scope=scope)

            z = tf.cond(tf.cast(train, tf.bool), lambda: bn_train, lambda: bn_infer)
            return z

        # deep层把维度拉开，bx(f*e)
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])

        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                                                            weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                l2_reg), scope='dense_%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = BN(deep_inputs, train=is_train, scope='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[
                    i])
        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   scope='deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("out"):
        # bias
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # predict输出
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
           l2_reg * tf.nn.l2_loss(FM_W) + \
           l2_reg * tf.nn.l2_loss(FM_V)

    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }

    # EVAL输出
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # train输出
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main(_):
    # 时间标记ckpt地址
    FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    if FLAGS.clear_existing_model:
        shutil.rmtree(FLAGS.model_dir)

    params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    # 数据加载
    if FLAGS.task_type == 'train':
        path_train = os.path.join(FLAGS.data_dir, 'train.csv')
        path_valid = os.path.join(FLAGS.data_dir, 'validation.csv')
    elif FLAGS.task_type == 'eval':
        path_valid = os.path.join(FLAGS.data_dir, 'validation.csv')
    elif FLAGS.task_type == 'test':
        path_test = os.path.join(FLAGS.data_dir, 'test.csv')
    else:
        print("not data requested")

    # 模型区
    config = tf.estimator.RunConfig().replace(model_dir=FLAGS.model_dir,
                                              save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    DeepFM = tf.estimator.Estimator(model_fn=my_model,
                                    config=config,
                                    params=params)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(path_train, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(path_valid, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)

    elif FLAGS.task_type == 'eval':
        DeepFM.evaluate(
            input_fn=lambda: input_fn(path_valid, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'test':
        preds = DeepFM.predict(
            input_fn=lambda: input_fn(path_test, num_epochs=1, batch_size=FLAGS.batch_size),
            predict_keys="prob")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        DeepFM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

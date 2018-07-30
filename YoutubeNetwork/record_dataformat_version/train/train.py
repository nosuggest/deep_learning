import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import read_tfrecords
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class Args():
    is_training = True
    embedding_size = 256
    brand_list = None
    msort_list = None
    item_count = -1
    brand_count = -1
    msort_count = -1
    lr = 1


if __name__ == '__main__':
    with open('/home/slade/Youtube/record/data/args_data.pkl', 'rb') as f:
        item_key, brand_key, msort_key = pickle.load(f)
        brand_list = pickle.load(f)
        msort_list = pickle.load(f)
        item_count, brand_count, msort_count, example_count = pickle.load(f)

    checkpoint_dir = '/home/slade/Youtube/record/data/save_path/ckpt'
    args = Args()
    args.brand_list = brand_list
    args.msort_list = msort_list
    args.item_count = item_count
    args.brand_count = brand_count
    args.msort_count = msort_count
    args.checkpoint_dir = checkpoint_dir

    epoch = 3
    batch_size = 32

    padded_shapes = ([None], [None], [None], [], [], [])

    input = read_tfrecords.batched_data('/home/slade/Youtube/record/data/train.tfrecords',
                                        read_tfrecords.single_example_parser, batch_size,
                                        ([None], [None], [None], [], [], []), epoch)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = Model(args)
        # print(sess.run(input))
        model.build_model(input)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if args.is_training:
            lr = 1.0
            all_loss = 0.0
            start_time = time.time()
            try:
                while not coord.should_stop():
                    # print('input',sess.run(input))
                    loss, _ = sess.run([model.loss, model.train_op])
                    all_loss += loss
                    # print('loss is %s\t' %loss)
                    if model.global_step.eval() % 1000 == 0:
                        model.save(sess, checkpoint_dir)
                        print('Global_step %d\tTrain_loss: %.4f' %
                              (model.global_step.eval(),
                               all_loss / model.global_step.eval()))

            except tf.errors.OutOfRangeError:
                print(' DONE\tCost time: %.2f' %
                      (time.time() - start_time))
                model.save(sess, checkpoint_dir)
                # print("done training")
            finally:
                coord.request_stop()
            coord.join(threads)

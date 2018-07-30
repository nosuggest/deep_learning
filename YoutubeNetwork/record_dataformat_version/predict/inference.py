import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import read_inference
from inference_model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class Args():
    is_training = False
    embedding_size = 256
    brand_list = None
    msort_list = None
    item_count = -1
    brand_count = -1
    msort_count = -1
    checkpoint_dir = None
    lr = 1


if __name__ == '__main__':
    with open('/home/slade/Youtube/record/data/args_data.pkl', 'rb') as f:
        item_key, brand_key, msort_key = pickle.load(f)
        brand_list = pickle.load(f)
        msort_list = pickle.load(f)
        item_count, brand_count, msort_count, example_count = pickle.load(f)

    # init args
    # recall_skn_idx=[0,9,8]
    checkpoint_dir = '/home/slade/Youtube/record/data/save_path/ckpt'
    args = Args()
    args.brand_list = brand_list
    args.msort_list = msort_list
    args.item_count = item_count
    args.brand_count = brand_count
    args.msort_count = msort_count
    args.checkpoint_dir = checkpoint_dir

    epoch = 1
    batch_size = 32

    padded_shapes = ([None], [], [], [])

    input = read_inference.batched_data('/home/slade/Youtube/record/data/inference.tfrecords',
                                        read_inference.single_example_parser, batch_size,
                                        ([None], [], [], []), epoch)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = Model(args)
        # print(sess.run(input))
        # model.build_model(input,recall_skn_idx)
        model.build_model(input)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model.restore(sess, checkpoint_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        out_file_skn = open("/home/slade/Youtube/record/data/inference_skn.txt", "w")
        start_time = time.time()
        try:
            while not coord.should_stop():
                # print('input', sess.run(input))

                output, uid = sess.run([model.output, model.u])
                pre_index = np.argsort(-output, axis=1)[:, 0:20]
                # recall
                # pre_index=recall_skn_idx[pre_index]
                # print('pre_index',pre_index)
                for y in range(len(pre_index)):
                    out_file_skn.write(str(uid[y]))
                    pre_skn = pre_index[y]
                    print(pre_skn)
                    for k in pre_skn:
                        out_file_skn.write("\t%i" % item_key[k])
                    out_file_skn.write("\n")


        except tf.errors.OutOfRangeError:
            print(' DONE\tCost time: %.2f' %
                  (time.time() - start_time))

        finally:
            coord.request_stop()
        coord.join(threads)

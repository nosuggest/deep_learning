import random
import pickle
import numpy as np
import tensorflow as tf
import time

random.seed(1234)


def write_records(uid, sl, last, hist, sub_sample, mask, f):
    frame_hist = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), hist))
    frame_sub_sample = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sub_sample))
    frame_mask = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), mask))
    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
                'sl': tf.train.Feature(int64_list=tf.train.Int64List(value=[sl])),
                'last': tf.train.Feature(int64_list=tf.train.Int64List(value=[last]))

            }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'hist': tf.train.FeatureList(feature=frame_hist),
            'sub_sample': tf.train.FeatureList(feature=frame_sub_sample),
            'mask': tf.train.FeatureList(feature=frame_mask)

        })
    )
    f.write(example.SerializeToString())


def write_inf_records(uid, sl, last, hist, f):
    frame_hist = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), hist))
    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
            'sl': tf.train.Feature(int64_list=tf.train.Int64List(value=[sl])),
            'last': tf.train.Feature(int64_list=tf.train.Int64List(value=[last]))

        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'hist': tf.train.FeatureList(feature=frame_hist)
        })
    )
    f.write(example.SerializeToString())


def generate_tfrecord(tfrecord_file, userclick_data, item_count, infer_tfrecord):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    infer_writer = tf.python_io.TFRecordWriter(infer_tfrecord)

    for UId, hist in userclick_data.groupby('UId'):
        pos_list = hist['ItemId'].tolist()
        if (len(pos_list) > 20):
            infer_list = pos_list[-20:]
        else:
            infer_list = pos_list
        if len(pos_list) < 3:
            continue

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(20 * len(pos_list))]
        neg_list = np.array(neg_list)
        for i in range(1, len(pos_list)):
            index = np.random.randint(len(neg_list), size=20)
            hist = pos_list[:i]
            sl = len(hist)
            last = pos_list[i - 1]
            if i != len(pos_list):
                cur_click = pos_list[i]
                neg_click = list(neg_list[index])
                sub_sample = [cur_click] + neg_click
                sub_sample_size = len(sub_sample)
                mask = np.zeros(sub_sample_size, np.int64)
                mask[0] = 1
                write_records(UId, sl, last, hist, sub_sample, mask, writer)

        inf_sl = len(infer_list)
        inf_last = infer_list[-1]
        write_inf_records(UId, inf_sl, inf_last, infer_list, infer_writer)
    infer_writer.close()
    writer.close()


start_time = time.time()
with open('/home/slade/Youtube/record/data/user_seq.pkl', 'rb') as f:
    userclick_data = pickle.load(f)
item_count = len(userclick_data['ItemId'].unique().tolist())
print((item_count))
tfrecord_file = '/home/slade/Youtube/record/data/train.tfrecords'
infer_tfrecord = '/home/slade/Youtube/record/data/inference.tfrecords'
generate_tfrecord(tfrecord_file, userclick_data, item_count, infer_tfrecord)
print(' DONE\tCost time: %.2f' % (time.time() - start_time))

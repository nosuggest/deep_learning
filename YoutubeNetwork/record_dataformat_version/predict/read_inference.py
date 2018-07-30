import tensorflow as tf
def single_example_parser(serialized_example):
    context_features = {

        "uid": tf.FixedLenFeature([], dtype=tf.int64),
        "sl": tf.FixedLenFeature([], dtype=tf.int64),
        "last": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "hist": tf.FixedLenSequenceFeature([], dtype=tf.int64),

    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    uid = context_parsed['uid']
    sl = context_parsed['sl']
    last = context_parsed['last']
    sequences = sequence_parsed['hist']

    return sequences, uid,sl,last

def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, num_epochs=1, buffer_size=1000):
    dataset = tf.contrib.data.TFRecordDataset(tfrecord_filename)\
        .map(single_example_parser)\
        .padded_batch(batch_size, padded_shapes=padded_shapes)\
        .repeat(num_epochs)
    return dataset.make_one_shot_iterator().get_next()
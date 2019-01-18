import numpy as np
import tensorflow as tf
import sys
from build_data import load_data

'''
author : taowei.sha(slade sha)
time : 18.07.27
'''


class Args():
    feature_sizes = 100
    field_size = 15
    embedding_size = 256
    deep_layers = [512, 256, 128]
    epoch = 3
    batch_size = 64
    learning_rate = 1.0
    l2_reg_rate = 0.01
    checkpoint_dir = '/Users/slade/Documents/Code/ml/data/saver/ckpt'
    is_training = True
    # deep_activation = tf.nn.relu


class model():
    def __init__(self, args):
        self.feature_sizes = args.feature_sizes
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.deep_activation = tf.nn.relu
        self.weight = dict()
        self.checkpoint_dir = args.checkpoint_dir
        self.build_model()

    def build_model(self):
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
        self.label = tf.placeholder(tf.float32, shape=[None, None], name='label')

        # 特征向量化，类似原论文中的v
        self.weight['feature_weight'] = tf.Variable(
            tf.random_normal([self.feature_sizes, self.embedding_size], 0.0, 0.01),
            name='feature_weight')

        # 一次项中的w系数，类似原论文中的w
        self.weight['feature_first'] = tf.Variable(
            tf.random_normal([self.feature_sizes, 1], 0.0, 1.0),
            name='feature_first')

        num_layer = len(self.deep_layers)
        # deep网络初始input：把向量化后的特征进行拼接后带入模型，n个特征*embedding的长度
        input_size = self.field_size * self.embedding_size
        init_method = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        self.weight['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        self.weight['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(1, self.deep_layers[0])), dtype=np.float32
        )
        # 生成deep network里面每层的weight 和 bias
        if num_layer != 1:
            for i in range(1, num_layer):
                init_method = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weight['layer_' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)
                self.weight['bias_' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(1, self.deep_layers[i])),
                    dtype=np.float32)

        # deep部分output_size + 一次项output_size + 二次项output_size
        last_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size
        init_method = np.sqrt(np.sqrt(2.0 / (last_layer_size + 1)))
        # 生成最后一层的结果
        self.weight['last_layer'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(last_layer_size, 1)), dtype=np.float32)
        self.weight['last_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # embedding_part
        self.embedding_index = tf.nn.embedding_lookup(self.weight['feature_weight'],
                                                      self.feat_index)  # Batch*F*K
        self.embedding_part = tf.multiply(self.embedding_index,
                                          tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # [Batch*F*1] * [Batch*F*K] = [Batch*F*K],用到了broadcast的属性
        print('embedding_part:', self.embedding_part)
        # embedding_part: Tensor("Mul:0", shape=(?, 15, 256), dtype=float32)

        # first_order
        self.embedding_first = tf.nn.embedding_lookup(self.weight['feature_first'],
                                                      self.feat_index)  # bacth*F*1
        self.embedding_first = tf.multiply(self.embedding_first, tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        self.first_order = tf.reduce_sum(self.embedding_first, 2)
        print('first_order:', self.first_order)
        # first_order: Tensor("Sum:0", shape=(?, 15), dtype=float32)

        # second_order
        self.sum_second_order = tf.reduce_sum(self.embedding_part, 1)
        self.sum_second_order_square = tf.square(self.sum_second_order)
        print('sum_square_second_order:', self.sum_second_order_square)
        # sum_square_second_order: Tensor("Square:0", shape=(?, 256), dtype=float32)

        self.square_second_order = tf.square(self.embedding_part)
        self.square_second_order_sum = tf.reduce_sum(self.square_second_order, 1)
        print('square_sum_second_order:', self.square_second_order_sum)
        # square_sum_second_order: Tensor("Sum_2:0", shape=(?, 256), dtype=float32)

        # 1/2*((a+b)^2 - a^2 - b^2)=ab
        self.second_order = 0.5 * tf.subtract(self.sum_second_order_square, self.square_second_order_sum)

        self.fm_part = tf.concat([self.first_order, self.second_order], axis=1)
        print('fm_part:', self.fm_part)
        # fm_part: Tensor("concat:0", shape=(?, 271), dtype=float32)

        # deep part
        self.deep_embedding = tf.reshape(self.embedding_part, [-1, self.field_size * self.embedding_size])
        print('deep_embedding:', self.deep_embedding)

        for i in range(0, len(self.deep_layers)):
            self.deep_embedding = tf.add(tf.matmul(self.deep_embedding, self.weight["layer_%d" % i]),
                                         self.weight["bias_%d" % i])
            self.deep_embedding = self.deep_activation(self.deep_embedding)

        # concat
        din_all = tf.concat([self.fm_part, self.deep_embedding], axis=1)
        self.out = tf.add(tf.matmul(din_all, self.weight['last_layer']), self.weight['last_bias'])
        print('output:', self.out)

        # loss
        self.out = tf.nn.sigmoid(self.out)

        # loss = tf.losses.log_loss(label,out) 也行，看大家想不想自己了解一下loss的计算过程
        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        # 正则：sum(w^2)/2*l2_reg_rate
        # 这边只加了weight，有需要的可以加上bias部分
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["last_layer"])
        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["layer_%d" % i])

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])


if __name__ == '__main__':
    args = Args()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session(config=gpu_config) as sess:
        Model = model(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = Model.train(sess, X_index, X_value, y)
                    if j % 100 == 0:
                        print('the times of training is %d, and the loss is %s' % (j, loss))
                        Model.save(sess, args.checkpoint_dir)
        else:
            Model.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = Model.predict(sess, X_index, X_value)
                print(result)

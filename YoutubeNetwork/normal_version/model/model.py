import tensorflow as tf


class Model(object):
    def __init__(self, args):

        self.is_training = args.is_training
        self.embedding_size = args.embedding_size
        # self.basic_size=args.basic_size
        self.brand_list = args.brand_list
        self.msort_list = args.msort_list
        self.item_count = args.item_count
        self.brand_count = args.brand_count
        self.msort_count = args.msort_count
        self.build_model()

    def build_model(self):
        # placeholder
        # self.u = tf.placeholder(tf.int32, [None, ])  # user idx [B]
        self.hist_click = tf.placeholder(tf.int32, [None, None])  # history click[B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # history len [B]
        self.last_click = tf.placeholder(tf.int32, [None, 1])  # last click[B]
        # self.basic = tf.placeholder(tf.float32, [None, 4])  # user basic feature[B,basic_size]
        self.sub_sample = tf.placeholder(tf.int32, [None, None])  # soft layer (pos_clict,neg_list)[B,sub_size]
        self.y = tf.placeholder(tf.float32, [None, None])  # label one hot[B]
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float64, [])

        # emb variable wx+b
        item_emb_w = tf.get_variable("item_emb_w", [self.item_count, self.embedding_size])
        brand_emb_w = tf.get_variable("brand_emb_w", [self.brand_count, self.embedding_size])
        msort_emb_w = tf.get_variable("msort_emb_w", [self.msort_count, self.embedding_size])

        input_b = tf.get_variable("input_b", [self.item_count], initializer=tf.constant_initializer(0.0))

        brand_list = tf.convert_to_tensor(self.brand_list, dtype=tf.int32)
        msort_list = tf.convert_to_tensor(self.msort_list, dtype=tf.int32)

        # historty click including item brand and sort , concat as axis = 2
        hist_brand = tf.gather(brand_list, self.hist_click)
        hist_sort = tf.gather(msort_list, self.hist_click)
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_click),
                           tf.nn.embedding_lookup(brand_emb_w, hist_brand),
                           tf.nn.embedding_lookup(msort_emb_w, hist_sort)], axis=2)

        # historty mask only calculate the click action
        mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]
        h_emb *= mask  # [B,T,3*e]
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        hist = tf.div(hist,
                      tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, 3 * self.embedding_size]), tf.float32))  # [B,3*e]

        # last click including item brand and sort , concat as axis = 2
        last_b = tf.gather(brand_list, self.last_click)
        last_m = tf.gather(msort_list, self.last_click)
        last_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.last_click),
                              tf.nn.embedding_lookup(brand_emb_w, last_b),
                              tf.nn.embedding_lookup(msort_emb_w, last_m)], axis=-1)
        last_emb = tf.squeeze(last_emb, axis=1)

        # self.input = tf.concat([hist, last_emb, self.basic], axis=-1)
        self.input = tf.concat([hist, last_emb], axis=-1)

        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)
        layer_3 = tf.layers.dense(layer_2, 3 * self.embedding_size, activation=tf.nn.relu, name='f3')

        # softmax
        if self.is_training:

            # find brand and sort idx
            sam_b = tf.gather(brand_list, self.sub_sample)
            sam_m = tf.gather(msort_list, self.sub_sample)

            # get item/brand/sort embedding vector and concat them
            sample_b = tf.nn.embedding_lookup(input_b, self.sub_sample)  # [B,sample]
            sample_w = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.sub_sample),
                                  tf.nn.embedding_lookup(brand_emb_w, sam_b),
                                  tf.nn.embedding_lookup(msort_emb_w, sam_m)
                                  # tf.tile(tf.expand_dims(self.basic, 1), [1, tf.shape(sample_b)[1], 1])
                                  ], axis=2)  # [B,sample,3*e]

            user_v = tf.expand_dims(layer_3, 1)  # [B,1,3*e]
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])  # [B,3*e,sample]

            self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b

            # Step variable
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            self.yhat = tf.nn.softmax(self.logits)

            self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))

            trainable_params = tf.trainable_variables()
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)


        else:
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(brand_emb_w, brand_list),
                                 tf.nn.embedding_lookup(msort_emb_w, msort_list)],
                                axis=1)
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True) + input_b
            self.output = tf.nn.softmax(self.logits)

    def train(self, sess, uij, l, keep_prob):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.sub_sample: uij[1],
            self.y: uij[2],
            self.hist_click: uij[3],
            self.sl: uij[4],
            self.last_click: uij[6],
            self.lr: l,
            self.keep_prob: keep_prob
        })
        return loss

    def test(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            # self.basic: uij[3],
            self.hist_click: uij[1],
            self.sl: uij[2],
            self.last_click: uij[4],
            self.keep_prob: keep_prob
        })

    def eval(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            # self.basic: uij[3],
            self.hist_click: uij[1],
            self.sl: uij[2],
            self.last_click: uij[4],
            self.keep_prob: keep_prob
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

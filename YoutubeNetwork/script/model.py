
import tensorflow as tf

__method__ = 'Deep Neural Networks for YouTube Recommendations'
__author__ = 'sladesal'

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
        self.basic = tf.placeholder(tf.float32, [None, 24])  # user basic feature[B,basic_size]
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

        # last click including item brand and sort , concat as axis = 2
        last_b = tf.gather(brand_list, self.last_click)
        last_m = tf.gather(msort_list, self.last_click)
        last_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.last_click),
                              tf.nn.embedding_lookup(brand_emb_w, last_b),
                              tf.nn.embedding_lookup(msort_emb_w, last_m)], axis=-1)
        last_emb = tf.squeeze(last_emb, axis=1)

        with tf.variable_scope("user_hist_group"):
            for i in range(2):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    h_emb, stt_vec = multihead_attention(queries=h_emb,
                                                       queries_length=self.sl,
                                                       keys=h_emb,
                                                       keys_length=self.sl,
                                                       num_units=None,
                                                       num_heads=8,
                                                       dropout_rate=0,
                                                       is_training=self.is_training,
                                                       scope="self_attention"
                                                       )

                    ### Feed Forward
                    h_emb = feedforward(h_emb,
                                      num_units=[192 // 4, 192],
                                      scope="feed_forward")

        last_emb = tf.expand_dims(last_emb, 1)
        with tf.variable_scope("item_feature_group"):
            for i in range(2):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( vanilla attention)
                    last_emb, att_vec = multihead_attention(queries=last_emb,
                                                       queries_length=tf.ones_like(last_emb[:, 0, 0], dtype=tf.int32),
                                                       keys=h_emb,
                                                       keys_length=self.sl,
                                                       num_units=None,
                                                       num_heads=8,
                                                       dropout_rate=0,
                                                       is_training=self.is_training,
                                                       scope="vanilla_attention")

                    ## Feed Forward
                    last_emb = feedforward(last_emb,
                                      num_units=[192 // 4, 192],
                                      scope="feed_forward")

        last_emb = tf.reshape(last_emb, [-1, 192])
        # historty mask only calculate the click action
        # mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [B,T]
        # mask = tf.expand_dims(mask, -1)  # [B,T,1]
        # mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]
        # h_emb *= mask  # [B,T,3*e]
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        # hist = tf.div(hist,
        #               tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, 3 * self.embedding_size]), tf.float32))  # [B,3*e]



        self.input = tf.concat([hist, last_emb, self.basic], axis=-1)

        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 512, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 256, activation=tf.nn.relu, name='f2')
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
            # layer_3 = tf.expand_dims(layer_3, 1)
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(brand_emb_w, brand_list),
                                 tf.nn.embedding_lookup(msort_emb_w, msort_list)],
                                # tf.tile(self.basic, [tf.shape(item_emb_w)[0], 1])],
                                axis=1)  # [B,sample,3*e+4]
            # all_emb_1 = tf.tile(tf.expand_dims(all_emb, 0), [tf.shape(layer_3)[0], 1, 1])
            # all_emb_2 = tf.concat([all_emb_1, tf.tile(tf.expand_dims(self.basic, 1), [1, tf.shape(all_emb_1)[1], 1])], axis=2)

            # self.logits = tf.squeeze(tf.matmul(layer_3, all_emb_2, transpose_b=True) + input_b, axis=1)
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True) + input_b
            self.output = tf.nn.softmax(self.logits)

    def train(self, sess, uij, l, keep_prob):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            # self.u: uij[0],
            self.sub_sample: uij[1],
            self.y: uij[2],
            self.hist_click: uij[3],
            self.sl: uij[4],
            self.basic: uij[5],
            self.last_click: uij[6],
            self.lr: l,
            self.keep_prob: keep_prob
        })
        return loss

    def test(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            self.basic: uij[3],
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

def multihead_attention(queries,
                        queries_length,
                        keys,
                        keys_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # query-key score matrix
        # each big score matrix is then split into h score matrix with same size
        # w.r.t. different part of the feature
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality = Future blinding: No use, removed

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs, att_vec

def feedforward(inputs,
                num_units=[512, 192],
                scope="feedforward"):
    with tf.variable_scope(scope):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)
    return outputs

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs
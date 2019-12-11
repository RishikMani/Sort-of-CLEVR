import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Dropout, Conv2D, BatchNormalization

from util import log


class Model:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.ques_dim = config.ques_dim
        self.ans_dim = config.ans_dim
        self.question_token_to_idx = config.question_token_to_idx
        self.padding = config.padding
        self.loss = tf.Variable(0.0, name="loss")
        self.accuracy = tf.Variable(0.0, name="accuracy")
        self.img = tf.compat.v1.placeholder(
            shape=[None, self.img_size, self.img_size, 3],
            dtype=tf.float32,
            name="image"
        )
        self.ques = tf.compat.v1.placeholder(
            shape=[None, self.ques_dim],
            dtype=tf.float32,
            name="question"
        )
        self.ans = tf.compat.v1.placeholder(
            shape=[None, self.ans_dim],
            dtype=tf.int16,
            name="answer"
        )
        
        self.build()
        
    def build(self):
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=labels
            )

            # Classification accuracy
            correct_prediction = tf.equal(
                tf.math.argmax(logits, 1),
                tf.math.argmax(labels, 1)
            )
            accuracy = tf.reduce_mean(tf.cast(
                correct_prediction,
                dtype=tf.float32
                )
            )
            
            return tf.reduce_mean(loss), accuracy
            
        def rnn(question, scope="RNN"):
            with tf.compat.v1.variable_scope(scope) as scope:
                log.warning(scope)
                x = Embedding(len(self.question_token_to_idx), 300)(question)
                x = LSTM(128)(x)
                return x
                
        def concat_coor(o, i, d):
            o = tf.compat.v1.keras.backend.repeat_elements(o, 23, 0)
            coor = tf.tile(
                tf.expand_dims([float(int(i / d)) / d, (i % d) / d], axis=0),
                [self.batch_size * 23, 1]
            )
            o = tf.concat([o, tf.cast(coor, dtype=tf.float32)], axis=1)
            return o
        
        def g_theta(o_i, o_j, q, scope="g_theta", reuse=True):
            with tf.compat.v1.variable_scope(scope, reuse=reuse) as scope:
                if not reuse:
                    log.warning(scope.name)
                g_1 = Dense(256, activation=tf.nn.relu, name="g_1")(tf.concat(
                    [o_i, o_j, q],
                    axis=1
                    )
                )
                g_2 = Dense(256, activation=tf.nn.relu, name="g_2")(g_1)
                g_3 = Dense(256, activation=tf.nn.relu, name="g_3")(g_2)
                g_4 = Dense(256, activation=tf.nn.relu, name="g_4")(g_3)
                return g_4
                
        def cnn(image, ques, scope="CONV"):
            with tf.compat.v1.variable_scope(scope) as scope:
                log.warning(scope)
                conv_1 = Conv2D(
                    24,
                    kernel_size=5,
                    strides=3,
                    activation=tf.nn.relu,
                    padding=self.padding,
                    name="conv_1")(image)
                bn_1 = BatchNormalization(name="bn_1")(conv_1)
                conv_2 = Conv2D(
                    24,
                    kernel_size=5,
                    strides=3,
                    activation=tf.nn.relu,
                    padding=self.padding,
                    name="conv_2")(bn_1)
                bn_2 = BatchNormalization(name="bn_2")(conv_2)
                conv_3 = Conv2D(
                    24,
                    kernel_size=5,
                    strides=2,
                    activation=tf.nn.relu,
                    padding=self.padding,
                    name="conv_3")(bn_2)
                bn_3 = BatchNormalization(name="bn_3")(conv_3)
                conv_4 = Conv2D(
                    24,
                    kernel_size=5,
                    strides=2,
                    activation=tf.nn.relu,
                    padding=self.padding,
                    name="conv_4")(bn_3)
                bn_4 = BatchNormalization(name="bn_4")(conv_4)
                
                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                d = bn_4.get_shape().as_list()[1]
                all_g = []
                for i in range(d*d):
                    o_i = bn_4[:, int(i / d), int(i % d), :]
                    o_i = concat_coor(o_i, i, d)
                    for j in range(d*d):
                        o_j = bn_4[:, int(j / d), int(j % d), :]
                        o_j = concat_coor(o_j, j, d)
                        if i == 0 and j == 0:
                            g_i_j = g_theta(o_i, o_j, ques, reuse=False)
                        else:
                            g_i_j = g_theta(o_i, o_j, ques, reuse=True)
                        all_g.append(g_i_j)
                        
                all_g = tf.stack(all_g, axis=0)
                all_g = tf.reduce_mean(all_g, axis=0, name="all_g")
                return all_g
                
        def f_phi(g, scope="f_phi"):
            with tf.compat.v1.variable_scope(scope) as scope:
                log.warning(scope.name)
                fc_1 = Dense(256, activation=tf.nn.relu, name="fc_1")(g)
                fc_2 = Dense(256, activation=tf.nn.relu, name="fc_2")(fc_1)
                fc_2 = Dropout(rate=0.5)(fc_2)
                fc_3 = Dense(self.ans_dim, activation=None, name="fc_3")(fc_2)
                return fc_3
                
        _rnn = rnn(self.ques)
        _cnn = cnn(self.img, _rnn)
        logits = f_phi(_cnn, scope="f_phi")
        self.all_preds = tf.nn.softmax(tf.convert_to_tensor(logits))
        self.loss, self.accuracy = build_loss(logits, self.ans)
            
        tf.compat.v1.summary.scalar("accuracy", self.accuracy)
        tf.compat.v1.summary.scalar("cross_entropy", self.loss)
        log.warning("Successfully loaded the model.")

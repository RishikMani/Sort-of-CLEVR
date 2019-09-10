import tensorflow as tf

from util import log

from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.layers import Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, concatenate


class Model:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.img_size
        self.ques_dim = config.ques_dim
        self.ans_dim = config.ans_dim
        self.padding = config.padding

        self.img = tf.placeholder(
            name='img',
            dtype=tf.float32,
            shape=[self.batch_size * 10, self.img_size, self.img_size, 3]
        )
        self.q = tf.placeholder(
            name='ques',
            dtype=tf.float32,
            shape=[self.batch_size * 10, self.ques_dim]
        )
        self.ans = tf.placeholder(
            name='ans',
            dtype=tf.float32,
            shape=[self.batch_size * 10, self.ans_dim]
        )

        self.loss = tf.Variable(0.0, name='loss')
        self.accuracy = tf.Variable(0.0, name='accuracy')

        self.build()

    def build(self):
        # Classifier: takes images as input and outputs class label [B, m]
        def C(img, q, scope):
            with tf.compat.v1.variable_scope(scope) as scope:
                log.warning(scope.name)

                conv_1 = Conv2D(24, kernel_size=5, strides=3, activation=tf.nn.relu, padding=self.padding, name='conv_1')(img)
                bn_1 = BatchNormalization(name='bn_1')(conv_1)
                conv_2 = Conv2D(24, kernel_size=5, strides=3, activation=tf.nn.relu, padding=self.padding, name='conv_2')(bn_1)
                bn_2 = BatchNormalization(name='bn_2')(conv_2)
                conv_3 = Conv2D(24, kernel_size=5, strides=2, activation=tf.nn.relu, padding=self.padding, name='conv_3')(bn_2)
                bn_3 = BatchNormalization(name='bn_3')(conv_3)
                conv_4 = Conv2D(24, kernel_size=5, strides=2, activation=tf.nn.relu, padding=self.padding, name='conv_4')(bn_3)
                bn_4 = BatchNormalization(name='bn_4')(conv_4)
                flat = Flatten(name='flatten')(bn_4)
                conv_q = tf.concat([flat, q], axis=1)
                fc_1 = Dense(256, activation=tf.nn.relu, name='fc_1')(conv_q)
                fc_2 = Dense(256, activation=tf.nn.relu, name='fc_2')(fc_1)
                fc_2 = Dropout(rate=0.5)(fc_2)
                fc_3 = Dense(self.ans_dim, activation=None, name='fc_3')(fc_2)
                return fc_3
        
        # build loss and accuracy
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )

            # Classification accuracy
            correct_prediction = tf.equal(tf.math.argmax(logits, 1), tf.math.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy

        logits = C(self.img, self.q, scope='Classifier')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.ans)

        tf.compat.v1.summary.scalar("accuracy", self.accuracy)
        tf.compat.v1.summary.scalar("cross_entropy", self.loss)
        log.warning('Successfully loaded the model.')

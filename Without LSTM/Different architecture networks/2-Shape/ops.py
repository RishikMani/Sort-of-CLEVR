import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv2d(input, output_shape, is_train, activation_fn=tf.nn.relu,
           k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        activation = activation_fn(conv + biases)
        bn = tf.contrib.layers.batch_norm(activation, center=True, scale=True,
                                          decay=0.9, is_training=is_train,
                                          updates_collections=None)
    return bn


def fc(input, output_shape, activation_fn=tf.nn.relu, name="fc"):
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    return output

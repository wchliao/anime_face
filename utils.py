import tensorflow as tf


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, is_training=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum,
                updates_collections=None, 
                is_training=is_training,
                epsilon=self.epsilon, scale=True,
                scope=self.name)


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)


def linear(x, output_dim, scope=None, stddev=0.02, bias_init=0.0):
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or 'Linear'):
        w = tf.get_variable('weight', [shape[1], output_dim], tf.float32,
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_dim],
                initializer=tf.constant_initializer(bias_init))
        
        return tf.matmul(x,w) + b


def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bias_init=0.0, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k_h, k_w, x.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding='SAME')
        b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(bias_init))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv
    

def deconv2d(x, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bias_init=0.0, name='deconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k_h, k_w, output_shape[-1], x.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1])
        b = tf.get_variable('bias', [output_shape[-1]],
            initializer=tf.constant_initializer(bias_init))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv



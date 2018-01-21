import tensorflow as tf


def filter(height, width, _in, out):
    return tf.Variable(tf.truncated_normal([height, width, _in, out], dtype=tf.float32,
                                           stddev=1e-1), name='weights')


def biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.01, shape=[shape], dtype=tf.float32),
                       trainable=True, name=name)


def conv3(inputlayers, fil_in, fil_out, stride):
    conv = tf.nn.conv2d(inputlayers, filter(3, 3, fil_in, fil_out), strides=stride,padding="SAME")
    bias = biases(fil_out)
    conv_bias = tf.add(conv, bias)
    return tf.nn.relu(tf.layers.batch_normalization(conv_bias))

def conv1(inputlayers, fil_in, fil_out, stride):
    conv = tf.nn.conv2d(inputlayers, filter(1, 1, fil_in, fil_out),strides=stride, padding="SAME")
    bias = biases(fil_out)
    conv_bias = tf.add(conv, bias)
    return tf.nn.relu(tf.layers.batch_normalization(conv_bias))


def maxpool(input, size, stride):
    return tf.nn.max_pool(input, size, stride,padding="VALID")

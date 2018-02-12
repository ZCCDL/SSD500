import tensorflow as tf


def filter(height, width, _in, out):
    return tf.Variable(tf.truncated_normal([height, width, _in, out], dtype=tf.float32,
                                           stddev=0.1), name='weights',trainable=True)
    #return tf.Variable(tf.constant(0.01,shape=[height, width, _in, out], dtype=tf.float32), name='weights',trainable=True)



def biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.1, shape=[shape], dtype=tf.float32),
                       trainable=True, name=name)


def conv3(inputlayers, fil_in, fil_out, stride,name="conv",activation="RELU"):
    weights=filter(3, 3, fil_in, fil_out)
    conv = tf.nn.conv2d(inputlayers, weights, strides=stride,padding="SAME")
    bias = biases(fil_out)
    conv_bias = tf.add(conv, bias)
    tf.summary.histogram("_bias", bias)
    tf.summary.histogram("_weights", weights)

    if activation!="RELU":
        return conv_bias

    activations=tf.nn.relu(tf.layers.batch_normalization(conv_bias))
    tf.summary.histogram("_activations", activations)

    return activations


def conv1(inputlayers, fil_in, fil_out, stride,activation="RELU"):
    weights= filter(1, 1, fil_in, fil_out)
    conv = tf.nn.conv2d(inputlayers, weights ,strides=stride, padding="SAME")
    bias = biases(fil_out)
    conv_bias = tf.add(conv, bias)
    tf.summary.histogram("_bias", bias)
    tf.summary.histogram("_weights", weights)

    if activation!="RELU":
        return conv_bias

    activations=tf.nn.relu(tf.layers.batch_normalization(conv_bias))
    tf.summary.histogram("_activations", activations)

    return activations


def maxpool(input, size, stride):
    return tf.nn.max_pool(input, size, stride,padding="VALID")

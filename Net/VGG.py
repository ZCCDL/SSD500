from tensorflow import layers
import tensorflow as tf
from Net.common import conv3, maxpool
import numpy as np
from tensorflow.contrib import slim, layers

class VGG:
    def __init__(self):
        return

    def load(self, link,sess):
        print("Loading Weights ...")

        self.weights = np.load(link)
        print(self.graph.keys())
        print(self.weights.keys())
        for key in self.graph.keys():
            print(key)
            sess.run(tf.assign(self.graph[key],self.weights[key],validate_shape=True))

    def inference(self, input):
        print(input.get_shape())
        self.graph={}

        tf.summary.image('input', input, max_outputs=1)
        with tf.variable_scope("conv_1"):

            conv1_1, conv1_1_W, conv1_1_b = conv3(input, 3, 64, [1, 1, 1, 1], "conv1_1")
            conv1_2, conv1_2_W, conv1_2_b = conv3(conv1_1, 64, 64, [1, 1, 1, 1], "conv1_2")

            self.graph["conv1_1_W"]=conv1_1_W
            self.graph["conv1_1_b"]=conv1_1_b
            self.graph["conv1_2_W"]=conv1_2_W
            self.graph["conv1_2_b"]=conv1_2_b

            pool1 = maxpool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1])

            print(pool1.get_shape())

        with tf.variable_scope("conv_2"):
            conv2_1, conv2_1_W, conv2_1_b = conv3(pool1, 64, 128, [1, 1, 1, 1], "conv2_1")
            conv2_2, conv2_2_W, conv2_2_b = conv3(conv2_1, 128, 128, [1, 1, 1, 1], "conv2_2")

            pool2 = maxpool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool2.get_shape())

            self.graph["conv2_1_W"] = conv2_1_W
            self.graph["conv2_2_W"] = conv2_2_W
            self.graph["conv2_1_b"] = conv2_1_b
            self.graph["conv2_2_b"] = conv2_2_b

        with tf.variable_scope("conv_3"):
            conv3_1, conv3_1_W, conv3_1_b = conv3(pool2, 128, 256, [1, 1, 1, 1], "conv3_1")
            conv3_2, conv3_2_W, conv3_2_b = conv3(conv3_1, 256, 256, [1, 1, 1, 1], "conv3_2")
            conv3_3, conv3_3_W, conv3_3_b = conv3(conv3_2, 256, 256, [1, 1, 1, 1], "conv3_3")

            pool3 = maxpool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool3.get_shape())

            self.graph["conv3_1_W"] = conv3_1_W
            self.graph["conv3_2_W"] = conv3_2_W
            self.graph["conv3_3_W"] = conv3_3_W
            self.graph["conv3_1_b"] = conv3_1_b
            self.graph["conv3_2_b"] = conv3_2_b
            self.graph["conv3_3_b"] = conv3_3_b

        with tf.variable_scope("conv_4"):
            conv4_1, conv4_1_W, conv4_1_b = conv3(pool3, 256, 512, [1, 1, 1, 1], "conv_4_1")
            conv4_2, conv4_2_W, conv4_2_b = conv3(conv4_1, 512, 512, [1, 1, 1, 1], "conv_4_2")
            conv4_3, conv4_3_W, conv4_3_b = conv3(conv4_2, 512, 512, [1, 1, 1, 1], "conv_4_3")

            print("4_3=", conv4_3.get_shape())

            pool4 = maxpool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool4.get_shape())

            self.graph["conv4_1_W"] = conv4_1_W
            self.graph["conv4_2_W"] = conv4_2_W
            self.graph["conv4_3_W"] = conv4_3_W
            self.graph["conv4_1_b"] = conv4_1_b
            self.graph["conv4_2_b"] = conv4_2_b
            self.graph["conv4_3_b"] = conv4_3_b


        with tf.variable_scope("conv_5"):
            conv5_1, conv5_1_W, conv5_1_b = conv3(pool4, 512, 512, [1, 1, 1, 1], "conv5_1")
            conv5_2, conv5_2_W, conv5_2_b = conv3(conv5_1, 512, 512, [1, 1, 1, 1], "conv5_2")
            conv5_3, conv5_3_W, conv5_3_b = conv3(conv5_2, 512, 512, [1, 1, 1, 1], "conv5_3")
            print("conv_5_3=", conv5_3.get_shape())

            pool5 = maxpool(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1])


            print("pool5=", pool5.get_shape())

            self.graph["conv5_1_W"] = conv5_1_W
            self.graph["conv5_2_W"] = conv5_2_W
            self.graph["conv5_3_W"] = conv5_3_W
            self.graph["conv5_1_b"] = conv5_1_b
            self.graph["conv5_2_b"] = conv5_2_b
            self.graph["conv5_3_b"] = conv5_3_b
        with tf.variable_scope("Dilation"):

            batchnorm = tf.layers.batch_normalization(pool5)
            initializer = layers.xavier_initializer_conv2d()

            dilation = slim.convolution2d_transpose(inputs=batchnorm, num_outputs=512, kernel_size=3,stride=2,padding='SAME',
                                 weights_initializer=initializer, biases_initializer=initializer,
                                 activation_fn=tf.nn.relu)
            print("dilation=", dilation.get_shape())


        return conv4_3, dilation

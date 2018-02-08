from tensorflow import layers
import tensorflow as tf
from Net.common import conv3,maxpool

class VGG:
    def __init__(self):
        return


    def inference(self, input):
        print(input.get_shape())

        with tf.variable_scope("conv_1"):
            conv1_1 = conv3(input, 3, 64, [1, 1, 1, 1])
            conv1_2 = conv3(conv1_1, 64, 64, [1, 1, 1, 1])

            pool1 = maxpool(conv1_2, [1,2, 2, 1], [1, 2, 2, 1])

            print(pool1.get_shape())


        with tf.variable_scope("conv_2"):
            conv2_1 = conv3(pool1, 64, 128, [1, 1, 1, 1])
            conv2_2 = conv3(conv2_1, 128, 128, [1, 1, 1, 1])

            pool2 = maxpool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool2.get_shape())

        with tf.variable_scope("conv_3"):
            conv3_1 = conv3(pool2, 128, 256, [1, 1, 1, 1])
            conv3_2 = conv3(conv3_1, 256, 256, [1, 1, 1, 1])
            conv3_3 = conv3(conv3_2, 256, 256, [1, 1, 1, 1])

            pool3 = maxpool(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool3.get_shape())

        with tf.variable_scope("conv_4"):

            conv4_1 = conv3(pool3, 256, 512, [1, 1, 1, 1])
            conv4_2 = conv3(conv4_1, 512, 512, [1, 1, 1, 1])
            conv4_3 = conv3(conv4_2, 512, 512, [1, 1, 1, 1])
            print("4_3=",conv4_3.get_shape())

            pool4 = maxpool(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1])
            print(pool4.get_shape())

        with tf.variable_scope("conv_5"):

            conv5_1 = conv3(pool4, 512, 512, [1, 1, 1, 1])
            conv5_2 = conv3(conv5_1, 512, 512, [1, 1, 1, 1])
            conv5_3 = conv3(conv5_2, 512, 512, [1, 1, 1, 1])
            pool5=maxpool(conv5_3,[1,2,2,1],[1,2,2,1])
            #print(pool5.get_shape())

        return conv4_3,pool5
from Net.common import conv1, conv3, maxpool
import tensorflow as tf
from Utils.anchors import getscales


class SSD500:
    def __init__(self,
                 conv4_3,
                 pool5,
                 num_classes,
                 aspect_ratio=[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 ):
        self.pool5 = pool5
        self.conv4_3 = conv4_3
        self.num_classes = num_classes
        self.aspect_ratio = aspect_ratio
        self.scales = getscales(7)
        self.inp_size = 512
        self.anchor_center = 0.5

        print(self.scales)

    def inference(self):
        conv4 = self.conv4_3

        featurelayers = {}

        featurelayers["conv4"] = conv4

        print("conv4=", conv4.get_shape())
        print("conv5=", self.pool5.get_shape())

        with tf.variable_scope("conv7"):
            # tf.layers.conv2d(self.pool5, filters=[1024], kernel_size=[3, 3], strides=[1, 1], dilation_rate=(6, 6),
            #                 activation='relu', padding="same", kernel_initializer='he_normal',kernel_regularizer)

            conv6, _, _ = conv3(self.pool5, 512, 1024, [1, 1, 1, 1])
            conv7, _, _ = conv1(conv6, 1024, 1024, [1, 1, 1, 1])
            print("conv7=", conv7.get_shape())

            featurelayers["conv7"] = conv7

        with tf.variable_scope("conv8"):
            conv8_1, _, _ = conv3(conv7, 1024, 256, [1, 1, 1, 1])
            conv8_2, _, _ = conv1(conv8_1, 256, 512, [1, 2, 2, 1])
            print("conv8=", conv8_2.get_shape())

            featurelayers["conv8"] = conv8_2

        with tf.variable_scope("conv9"):
            conv9_1, _, _ = conv3(conv8_2, 512, 128, [1, 1, 1, 1])
            conv9_2, _, _ = conv1(conv9_1, 128, 256, [1, 2, 2, 1])
            print("conv9=", conv9_2.get_shape())

            featurelayers["conv9"] = conv9_2

        with tf.variable_scope("conv10"):
            conv10_1, _, _ = conv3(conv9_2, 256, 128, [1, 1, 1, 1])
            conv10_2, _, _ = conv1(conv10_1, 128, 256, [1, 2, 2, 1])
            print("conv10=", conv10_2.get_shape())
            featurelayers["conv10"] = conv10_2

        with tf.variable_scope("conv11"):
            conv11_1, _, _ = conv3(conv10_2, 256, 128, [1, 1, 1, 1])
            conv11_2, _, _ = conv1(conv11_1, 128, 256, [1, 2, 2, 1])
            print("conv11=", conv11_2.get_shape())
            featurelayers["conv11"] = conv11_2

        with tf.variable_scope("conv12"):
            conv12_1, _, _ = conv3(conv11_2, 256, 128, [1, 1, 1, 1])
            conv12_2, _, _ = conv1(conv12_1, 128, 256, [1, 2, 2, 1])
            print("conv12=", conv12_2.get_shape())
            featurelayers["conv12"] = conv12_2
        # todo Dilate all the layers

        self.logits = {}
        self.regr = []
        self.pred = []
        self.cls = []
        with tf.variable_scope("reg-cls-head-4"):
            self.logits["conv4"] = self.final_layer(featurelayers["conv4"], 512, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-7"):
            self.logits["conv7"] = self.final_layer(featurelayers["conv7"], 1024, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-8"):
            self.logits["conv8"] = self.final_layer(featurelayers["conv8"], 512, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-9"):
            self.logits["conv9"] = self.final_layer(featurelayers["conv9"], 256, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-10"):
            self.logits["conv10"] = self.final_layer(featurelayers["conv10"], 256, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-11"):
            self.logits["conv11"] = self.final_layer(featurelayers["conv11"], 256, self.num_classes, 5)
        with tf.variable_scope("reg-cls-head-12"):
            self.logits["conv12"] = self.final_layer(featurelayers["conv12"], 256, self.num_classes, 5)

        return self.cls, self.regr, self.pred

    def final_layer(self, feature_map, in_filters, classes, num_ratios):
        out_filters_reg = (num_ratios) * 4
        out_filters_class = (num_ratios) * classes

        with tf.variable_scope("reg-head"):
            regression = conv3(feature_map, in_filters, out_filters_reg, [1, 1, 1, 1], activation=None)
            shape = regression.get_shape().as_list()[:-1] + [num_ratios, 4]
            regression = tf.reshape(regression, shape)
            print("regression=", regression.get_shape())
            self.regr.append(regression)

        with tf.variable_scope("class-head"):
            logits = conv3(feature_map, in_filters, out_filters_class, [1, 1, 1, 1], activation=None)
            shape = logits.get_shape().as_list()[:-1] + [num_ratios, classes]
            logits = tf.reshape(logits, shape)
            print("class_score=", logits.get_shape())
            self.cls.append(logits)
            # self.pred.append(tf.arg_max(tf.nn.softmax(logits),dimension=4))

        return logits, regression

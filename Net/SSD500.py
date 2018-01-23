from Net.common import conv1, conv3, maxpool
import tensorflow as tf
from Utils.anchors import getscales

class SSD500:
    def __init__(self, conv4_3, conv5_3, num_classes):
        self.conv5_3 = conv5_3
        self.conv4_3 = conv4_3
        self.num_classes = num_classes
        self.aspect_ratio = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5]]
        self.scales=getscales(6)
        self.inp_size=512
        self.anchor_center=0.5


        print(self.scales)
    def inference(self):
        conv4 = self.conv4_3

        featurelayers = {}

        featurelayers["conv4"] = conv4

        print("conv4=", conv4.get_shape())
        print("conv5=", self.conv5_3.get_shape())

        with tf.variable_scope("conv7"):
            conv6 = conv3(self.conv5_3, 512, 1024, [1, 1, 1, 1])
            conv7 = conv1(conv6, 1024, 1024, [1, 1, 1, 1])
            print("conv7=", conv7.get_shape())

            featurelayers["conv7"] = conv7

        with tf.variable_scope("conv8"):
            conv8_1 = conv3(conv7, 1024, 256, [1, 1, 1, 1])
            conv8_2 = conv1(conv8_1, 256, 512, [1, 2, 2, 1])
            print("conv8=", conv8_2.get_shape())

            featurelayers["conv8"] = conv8_2

        with tf.variable_scope("conv9"):
            conv9_1 = conv3(conv8_2, 512, 128, [1, 1, 1, 1])
            conv9_2 = conv1(conv9_1, 128, 256, [1, 2, 2, 1])
            print("conv9=", conv9_2.get_shape())

            featurelayers["conv9"] = conv9_2

        with tf.variable_scope("conv10"):
            conv10_1 = conv3(conv9_2, 256, 128, [1, 1, 1, 1])
            conv10_2 = conv1(conv10_1, 128, 256, [1, 2, 2, 1])
            print("conv10=", conv10_2.get_shape())
            featurelayers["conv10"] = conv10_2

        with tf.variable_scope("conv11"):
            conv11_1 = conv3(conv10_2, 256, 128, [1, 1, 1, 1])
            conv11_2 = conv1(conv11_1, 128, 256, [1, 2, 2, 1])
            print("conv11=", conv11_2.get_shape())
            featurelayers["conv11"] = conv11_2

        with tf.variable_scope("conv12"):
            conv12_1 = conv3(conv11_2, 256, 128, [1, 1, 1, 1])
            conv12_2 = conv1(conv12_1, 128, 256, [1, 2, 2, 1])
            print("conv12=", conv12_2.get_shape())
            featurelayers["conv12"] = conv12_2
        #todo here
        self.final_layer(featurelayers["conv4"], 512, self.num_classes, 5)
        self.final_layer(featurelayers["conv7"], 1024, self.num_classes, 5)
        self.final_layer(featurelayers["conv8"], 512, self.num_classes, 5)

        for i, feat_layer in enumerate(featurelayers.keys()):
            print(i)
            print(feat_layer)
            print("classes=", self.num_classes)

        return conv11_2

    def final_layer(self, feature_map, in_filters, classes, num_anchors):
        out_filters_reg = (num_anchors) * 4
        out_filters_class = (num_anchors) * classes

        regression = conv3(feature_map, in_filters, out_filters_reg, [1, 1, 1, 1])
        shape = regression.get_shape().as_list()[:-1] + [num_anchors, 4]
        regression = tf.reshape(regression, shape)
        print("regression=", regression.get_shape())

        logits = conv3(feature_map, in_filters, out_filters_class, [1, 1, 1, 1])
        shape = logits.get_shape().as_list()[:-1] + [num_anchors, classes]
        logits = tf.reshape(logits, shape)

        print("class_score=", logits.get_shape())

        return logits, regression

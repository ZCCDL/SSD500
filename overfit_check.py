import tensorflow as tf
from Net.VGG import VGG
from Net.SSD500 import SSD500
import sys
import numpy as np
import os
from Utils.anchor_match import fetch_data, encode_layers, prepare_batch
from Utils.loss import tloss,l1_loss
from Utils.visual_anchors import visual
from Utils.anchors import getscales
import os

batchsize = 2
imgdir = "JPEGImages/"
groundtruth = "Annotations/"
total_steps = 1000000
ckpt_dir = "ckpt/"
ckpt_steps = 5000
load = -1
gpu = 0.3
lr = 0.01

train_batch = tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])
labels_c = tf.placeholder(dtype=tf.int32, shape=[batchsize, 64, 64])
labels_r = tf.placeholder(dtype=tf.float32, shape=[batchsize, 64, 64,5,4])

param = 2
feature_map_sizes = [64, 32, 16, 8, 4, 2, 1]
numclasses = 2
aspect_ratio = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

vgg = VGG()
conv4_3, conv5_3 = vgg.inference(train_batch)
ssd500 = SSD500(conv4_3=conv4_3, pool5=conv5_3, num_classes=numclasses)
ssdinfer = ssd500.inference()

loss_op = tf.reduce_sum(l1_loss(labels_r,ssd500.regr[0]))

#tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=ssd500.conv4_3)

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter("summary/")
mergedsummary = tf.summary.merge_all()
with tf.Session(config=session_config) as sess:
    sess.run(init)
    i = 0

    img = np.random.randint(0,255,(batchsize, 512, 512, 3)).astype(np.float32)
    b_labels = np.random.randint(1,2,(batchsize, 64, 64,5,4), dtype=np.int32)
    print(b_labels)
    print(img)

    while True:
        i += 1
        _, loss = sess.run([train_step, loss_op], feed_dict={train_batch: img,
                                                             labels_r: b_labels,
                                                             })
        s = sess.run(mergedsummary, feed_dict={train_batch: img,
                                               labels_r: b_labels,
                                               })
        writer.add_summary(s, i)
        writer.add_graph(sess.graph)

        print("Step <", i, "> loss => ", loss)
        # print(logits.keys())

        # print(logits["conv4"][0].shape)

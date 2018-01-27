import tensorflow as tf
from Net.VGG import VGG
from Net.SSD500 import SSD500
import sys
import numpy as np
import os
from Utils.anchor_match import fetch_data, encode_layers
from Utils.loss import tloss

batchsize = 1
imgdir = "Utils/"
groundtruth = "Utils/"
total_steps = 1000000
ckpt_dir = "ckpt/"
ckpt_steps = 5000
load = -1
gpu = 0.3
lr = 1e-03

# print("--loadfrom;",sys.argv[1]," --ckptdir;",sys.argv[2]," --gpu",sys.argv[3]," --lr", sys.argv[4],"save",sys.argv[5])

# python main.py -1 ckpt 0.5 1e-4 100

feature_map_sizes = [64, 32, 16, 8, 4, 2, 1]
numclasses = 2
aspect_ratio = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)


train_batch = tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])
gt_cls64 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 64, 64, len(aspect_ratio)])
gt_reg64 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 64, 64, len(aspect_ratio), 4])
gt_cls32 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 32, 32, len(aspect_ratio)])
gt_reg32 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 32, 32, len(aspect_ratio), 4])
gt_cls16 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 16, 16, len(aspect_ratio)])
gt_reg16 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 16, 16, len(aspect_ratio), 4])
gt_cls8 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 8, 8, len(aspect_ratio)])
gt_reg8 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 8, 8, len(aspect_ratio), 4])
gt_cls4 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 4, 4, len(aspect_ratio)])
gt_reg4 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 4, 4, len(aspect_ratio), 4])
gt_cls2 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 2, 2, len(aspect_ratio)])
gt_reg2 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 2, 2, len(aspect_ratio), 4])
gt_cls1 = tf.placeholder(dtype=tf.int32, shape=[batchsize, 1, 1, len(aspect_ratio)])
gt_reg1 = tf.placeholder(dtype=tf.float32, shape=[batchsize, 1, 1, len(aspect_ratio), 4])

gt_cls = (gt_cls64, gt_cls32, gt_cls16, gt_cls8, gt_cls4, gt_cls2, gt_cls1)
gt_reg = (gt_reg64, gt_reg32, gt_reg16, gt_reg8, gt_reg4, gt_reg2, gt_reg1)


vgg = VGG()
conv4_3, conv5_3 = vgg.inference(train_batch)
ssd500 = SSD500(conv4_3=conv4_3, conv5_3=conv5_3, num_classes=numclasses)
ssdinfer = ssd500.inference()



loss_op = tloss(gt_cls, gt_reg, ssd500.cls, ssd500.regr)
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session(config=session_config) as sess:
    sess.run(init)
    while True:

        img, labels = fetch_data(batchsize, imgdir, groundtruth)
        # encode the ground truth into anchors
        batch_encodes_c, batch_encodes_r = encode_layers(batchsize, labels, feature_map_sizes, ssd500.scales, aspect_ratio)


        _,loss = sess.run([train_step,loss_op], feed_dict={train_batch: img,
                                         gt_cls: batch_encodes_c,
                                         gt_reg: batch_encodes_r
                                         })
        print("Total loss=", loss)
    # print(logits.keys())

    # print(logits["conv4"][0].shape)

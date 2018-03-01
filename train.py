import tensorflow as tf
from Net.VGG import VGG
from Net.SSD500 import SSD500
import sys
import numpy as np
import os
from Utils.anchor_match import fetch_data, prepare_batch, get_anchors_all_layers
from Utils.loss import tloss, ssd_losses
from Utils.visual_anchors import visual
from Utils.anchors import getscales
import os
from Utils.tfencode import tf_ssd_bboxes_encode

batchsize = 2
imgdir = "JPEGImages/"
groundtruth = "Annotations/"
total_steps = 1000000
ckpt_dir = "ckpt/"
ckpt_steps = 1000
load = -1
gpu = 0.3
lr = 0.00001
weights_file = "/Users/nomanshafqat/Downloads/vgg16_weights.npz"
# print("--loadfrom;",sys.argv[1]," --ckptdir;",sys.argv[2]," --gpu",sys.argv[3]," --lr", sys.argv[4],"save",sys.argv[5])

# python main.py -1 ckpt 0.5 1e-4 100
# print("--loadfrom;",sys.argv[1]," --ckptdir;",sys.argv[2]," --gpu",sys.argv[3]," --lr", sys.argv[4],"save",sys.argv[5])

# python main.py -1 ckpt 0.5 1e-4 100


load=int(sys.argv[1])
ckpt_dir=sys.argv[2]
gpu=float(sys.argv[3])
lr=float(sys.argv[4])
ckpt_steps=int(sys.argv[5])
batchsize=int(sys.argv[6])
imgdir=sys.argv[7]
groundtruth=sys.argv[8]
weights_file=sys.argv[9]
assert (os.path.exists(ckpt_dir))
assert (os.path.exists(imgdir))
assert (os.path.exists(groundtruth))

param = 2
feature_map_sizes = [64, 32, 16, 8, 4, 2, 1]
numclasses = 2
aspect_ratio = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

train_batch = tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])

# gt_cls = (gt_cls64, gt_cls32, gt_cls16, gt_cls8, gt_cls4, gt_cls2, gt_cls1)
# gt_reg = (gt_reg64, gt_reg32, gt_reg16, gt_reg8, gt_reg4, gt_reg2, gt_reg1)

vgg = VGG()
conv4_3, conv5_3 = vgg.inference(train_batch)
ssd500 = SSD500(conv4_3=conv4_3, pool5=conv5_3, num_classes=numclasses)
ssdinfer = ssd500.inference()
# vgg.load("dsf")


labels_plc = tf.placeholder(dtype=tf.int64, shape=[batchsize, None, 1], name="labels")
reg_bboxes_plc = tf.placeholder(dtype=tf.float32, shape=[batchsize, None, 4], name="bboxes")

anchors = get_anchors_all_layers(feature_map_sizes, ssd500.scales, aspect_ratio)

gt_cls, gt_reg, gy_reg = tf_ssd_bboxes_encode(labels_plc, reg_bboxes_plc, anchors, numclasses)
# gt_cls[0]=tf.stack((gt_cls[0],gt_cls[0]),axis=0)
loss_op = tloss(ssd500.cls, ssd500.regr, gt_cls, gt_reg)

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter("summary/")
mergedsummary = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session(config=session_config) as sess:
    sess.run(init)

    if load == -1:
        vgg.load(weights_file, sess)

    writer.add_graph(sess.graph)

    start = 0
    if load > 0:
        print("Restoring", load, ".ckpt.....")
        saver.restore(sess, os.path.join(ckpt_dir, str(load)))
        start = load
    while True:

        img, labels, bboxes = prepare_batch(imgdir, groundtruth, batchsize)

        print(img[0].shape, end="")
        # print(img)
        # encode the ground truth into anchors
        # batch_encodes_c, batch_encodes_r, batch_encodes_iou = encode_layers(batchsize, labels, feature_map_sizes,
        #                                                                     ssd500.scales, aspect_ratio)

        # batch_encodes_c[0][batch_encodes_c[0] > 0] = 1
        # batch_encodes_r[0][batch_encodes_r[0] > 0] = 1

        # visual(img[0],batch_encodes_r[0],batch_encodes_iou,aspect_ratio,getscales(7))

        # img=np.zeros((2,512,512,3),dtype=np.float32)

        #print(labels)
        #print(bboxes)
        # labels = [
        #     [[1], [1], [1]],
        #     [[1], [0], [0]]]
        # bboxes = [
        #     [[1.0, 1, 5, 6], [1, 1, 55, 66],[0, 0, 0, 0]],
        #     [[1., 4., 7.7, 7.7], [0, 0, 0, 0],[0, 0, 0, 0]]
        # ]
        # print(bbox78)

        _, loss = sess.run([train_step, loss_op], feed_dict={train_batch: img,
                                                             reg_bboxes_plc: bboxes,
                                                             labels_plc: labels,
                                                             })

        if start % 100 == 0:
            s = sess.run(mergedsummary, feed_dict={train_batch: img,
                                                   reg_bboxes_plc: bboxes,
                                                   labels_plc: labels,
                                                   })
            writer.add_summary(s, start)
            print("writing summary")

        print("Step <", start, "> loss => ", loss)
        if start % ckpt_steps == 0 and start != ckpt_steps:
            print("saving checkpoint ", str(start), ".ckpt.....")

            save_path = saver.save(sess, os.path.join(ckpt_dir, str(start)))

        start += 1

        # print(logits.keys())

        # print(logits["conv4"][0].shape)

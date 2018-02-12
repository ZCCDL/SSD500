from Net.SSD500 import SSD500
from Net.VGG import VGG
import cv2
import numpy
import tensorflow as tf
import os
ckpt_dir='ckpt'
numclasses = 2
batchsize=2
gpu=0.5
load=-1
test_batch_tensor = tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)



vgg = VGG()
conv4_3, conv5_3 = vgg.inference(test_batch_tensor)
ssd500 = SSD500(conv4_3=conv4_3, pool5=conv5_3, num_classes=numclasses)
ssdinfer = ssd500.inference()

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session(config=session_config) as sess:
    sess.run(init)
    if load > 0:
        print("Restoring", load, ".ckpt.....")
        saver.restore(sess, os.path.join(ckpt_dir, str(load)))
        start = load
    while True:
        img = cv2.imread("JPEGImages/frame540.jpg")
        img = cv2.resize(img, (512, 512))
        test_batch = numpy.stack((img, img), axis=0)
        print(test_batch.shape)
        cls_logits,regr,pred=sess.run(ssdinfer,feed_dict={test_batch_tensor:test_batch})
        print(pred[5])

        print(pred[5].shape)
        #TODO Decode anchors back



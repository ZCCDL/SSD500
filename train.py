
import tensorflow as tf
from Net.VGG import VGG
from Net.SSD500 import SSD500
import sys
import numpy as np
import os
batchsize=4
imgdir="DS"
groundtruth="GT"
total_steps=1000000
ckpt_dir="ckpt/"
ckpt_steps=5000
load=-1
gpu=0.5
lr=1e-04

#print("--loadfrom;",sys.argv[1]," --ckptdir;",sys.argv[2]," --gpu",sys.argv[3]," --lr", sys.argv[4],"save",sys.argv[5])

# python main.py -1 ckpt 0.5 1e-4 100






vgg=VGG()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

train_batch=tf.placeholder(dtype=tf.float32, shape=[batchsize, 512, 512, 3])


conv4_3,conv5_3=vgg.inference(train_batch)
ssd500=SSD500(conv4_3=conv4_3,conv5_3=conv5_3,num_classes=7)
ssdinfer=ssd500.inference()





init = tf.global_variables_initializer()



with tf.Session(config=session_config) as sess:
    sess.run(init)
    ans=sess.run(ssdinfer,feed_dict={train_batch:np.ones((batchsize,512,512,3))})
    print(ans)





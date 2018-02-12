import tensorflow as tf
import numpy as np
import cv2
batchsize = 2
imgdir = "/Users/nomanshafqat/Google Drive/Dataset/Aerial Labeling/Refined_output/JPEGImages/"
groundtruth = "/Users/nomanshafqat/Google Drive/Dataset/Aerial Labeling/Refined_output/Annotations/"
total_steps = 1000000
ckpt_dir = "ckpt/"
ckpt_steps = 5000
load = -1
gpu = 0.3
lr = 1e-04

param=1
feature_map_sizes = [64, 32, 16, 8, 4, 2, 1]
numclasses = 2
aspect_ratio = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]

from Utils.anchor_match import prepare_batch, encode_layers
from Utils.anchors import getscales
from Utils.visual_anchors import visual

img, labels = prepare_batch(batchsize, imgdir, groundtruth)
# encode the ground truth into anchors
batch_encodes_c, batch_encodes_r, batch_encodes_iou = encode_layers(batchsize, labels,
                                                                    feature_map_sizes, getscales(7),
                                                                    aspect_ratio)


visual(img, batch_encodes_r, batch_encodes_iou, aspect_ratio, getscales(7))
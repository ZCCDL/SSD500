import numpy as np
import math
import cv2
import random


def getscales(m):
    scales = []
    for k in range(1, m + 1):
        min = 0.02
        max = 0.90
        scales.append((min + (((max - min) / (m - 1)) * (k - 1))))
    #print("Scales=",scales)
    return scales


def gen_anchor_for_1_layer(feature_map_size, scale,aspect_ratios,img_size=512 ):

    y,x=np.mgrid[0:feature_map_size, 0:feature_map_size]
    y=(y.astype(np.float32)+0.5)*(img_size/feature_map_size)/img_size
    x=(x.astype(np.float32)+0.5)*(img_size/feature_map_size)/img_size
    #print(x,y)

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors=len(aspect_ratios)

    w = np.zeros((num_anchors, ), dtype=np.float32)
    h = np.zeros((num_anchors, ), dtype=np.float32)

    for i,a in enumerate(aspect_ratios):
        w[i] = math.sqrt(a) * scale
        h[i] = scale / math.sqrt(a)

    #print(w, h, "a=", a)

    return x,y,w,h


#gen_anchor_for_1_layer(64,getscales(7)[0],[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0])


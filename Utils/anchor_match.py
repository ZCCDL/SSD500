from Utils.anchors import gen_anchor_for_1_layer, getscales
from Utils.read_data import read_frame, convert_labels_to_xyhw
from Utils.iou import bb_intersection_over_union
import numpy as np
import cv2
import os
import random
import math
import tensorflow as tf


#from Utils.encode_layer import encode_labels_1_layer


def encode_labels_1_layer(labels, anchors):
    ax = anchors[0]
    ay = anchors[1]
    aw = anchors[2]
    ah = anchors[3]
    bboxes = np.zeros((ax.shape[0], ax.shape[1], len(aw), 4), dtype=np.float32)
    ious = np.zeros((ax.shape[0], ax.shape[1], len(aw)), dtype=np.float32)
    en_labels = np.zeros((ax.shape[0], ax.shape[1], len(aw)), dtype=np.int64)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            for k in range(len(aw)):

                # anchor box x,w,w,h
                d_cx = ax[i][j][0]
                d_cy = ay[i][j][0]
                d_w = aw[k]
                d_h = ah[k]

                # bbox xmin,ymin,xmax,ymax
                a_xmin = d_cx - d_w
                a_xmax = d_cx + d_w
                a_ymin = d_cy - d_h
                a_ymax = d_cy + d_h
                if a_xmin<0:
                    a_xmin=0
                if a_ymin<0:
                    a_ymin=0
                if a_xmax>512:
                    a_xmax=512
                if a_ymax>512:
                    a_ymax=512

                # print(axa)
                # print(ax[i][j][0])

                for box in labels[0][1][2]:
                    l, g_cx, g_cy, g_w, g_h = box
                    g_xmin = g_cx - g_w
                    g_xmax = g_cx + g_w
                    g_ymin = g_cy - g_h
                    g_ymax = g_cy + g_h

                    iou = bb_intersection_over_union([g_xmin, g_ymin, g_xmax, g_ymax], [a_xmin, a_ymin, a_xmax, a_ymax])
                    # print("iou=", iou)

                    if iou > ious[i][j][k] and iou > 0.4:
                        ious[i][j][k] = iou

                        cx = (g_cx - d_cx) / d_w
                        cy = (g_cy - d_cy) / d_h
                        h = math.log(g_h / d_h)
                        w = math.log(g_w / d_w)
                        #print(cx)
                        #cx = d_cx
                        #cy = d_cy
                        #h = d_h
                        #w = d_w

                        bboxes[i][j][k][0] = cx
                        bboxes[i][j][k][1] = cy
                        bboxes[i][j][k][2] = w
                        bboxes[i][j][k][3] = h

                        en_labels[i][j][k] = 1
                        #print(i,j,cx,cy,w,h ,iou)
                        # todo convert labels to numbers

                        # print(axa,aya,axb,ayb)
    # todo draw to test
    return bboxes, en_labels, ious



def encode_layers(batchsize, batch_labels, featuremap_widths, scales, ratios):
    batch_encodes_c = []
    batch_encodes_r = []
    batch_encodes_iou = []

    for i, width in enumerate(featuremap_widths):
        encoded_bbox = []
        encoded_class = []
        encoded_iou = []

        anchors = gen_anchor_for_1_layer(width, scales[i], ratios)

        for labels in batch_labels:

            box, encoded_labels, ious = encode_labels_1_layer(labels, anchors)
            encoded_bbox.append(box)
            encoded_class.append(encoded_labels)
            encoded_iou.append(ious)

        batch_encodes_c.append(np.array(encoded_class))
        batch_encodes_r.append(np.array(encoded_bbox))
        batch_encodes_iou.append(np.array(encoded_iou))
        # print(np.array(encoded_class).shape)
        # print("test", np.array(encoded_local).shape)

    return batch_encodes_c, batch_encodes_r, batch_encodes_iou


def fetch_data(batch_size, img_dir, ann_dir):
    batch_labels = []
    batch_img = []
    filenames = os.listdir(img_dir)
    while True:
        random.shuffle(filenames)
        for filename in filenames:

            if not filename.__contains__("jpg"):
                continue

            print(filename, end=" ")
            img, labels = read_frame(filename, img_dir, ann_dir)
            img = cv2.resize(img, (512, 512))

            batch_labels.append(convert_labels_to_xyhw(labels))
            batch_img.append(img)

            if len(batch_img) == batch_size:
                yield batch_img, batch_labels
                batch_labels = []
                batch_img = []


def prepare_batch(img_dir, ground_truth_dir, batch_size):
    batch = fetch_data(img_dir, ground_truth_dir, batch_size)

    for image, labels in batch:
        # print(image)
        # print(labels)
        return image, labels

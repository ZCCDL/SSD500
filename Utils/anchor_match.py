
from Utils.anchors import gen_anchor_for_1_layer,getscales
from Utils.read_data import read_frame,convert_labels_to_xyhw
from Utils.iou import bb_intersection_over_union
import numpy as np
import cv2
import tensorflow as tf
def encode_labels_1_layer(labels,anchors):


    ax = anchors[0]
    ay = anchors[1]
    aw = anchors[2]
    ah = anchors[3]
    localization = np.zeros((ax.shape[0], ax.shape[1], len(aw), 4),dtype=np.float32)
    encoded_label = np.zeros((ax.shape[0], ax.shape[1], len(aw)),dtype=np.int64)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            for k in range(len(aw)):
                axa=ax[i][j][0]-aw[k]
                aya=ay[i][j][0]-ah[k]

                axb=ax[i][j][0]+aw[k]
                ayb=ay[i][j][0]+ah[k]
                #print(axa)
                #print(ax[i][j][0])

                for box in labels[0][1][2]:
                    l, x, y, w, h = box
                    xa = x - w
                    xb = x + w
                    ya = y - h
                    yb = y + h

                    iou=bb_intersection_over_union([xa,xb,ya,yb],[axa,axb,aya,ayb])
                    #print("iou=", iou)

                    if iou>0.5:
                        localization[i][j][k][0]=x
                        localization[i][j][k][1]=y
                        localization[i][j][k][2]=w
                        localization[i][j][k][3]=h
                        #todo convert labels to numbers
                        encoded_label[i][j][k]=1
                        #print("iou=",iou)
                        #print(N)
                    else:
                        localization[i][j][k][0] = axa
                        localization[i][j][k][1] = axb
                        localization[i][j][k][2] = aya
                        localization[i][j][k][3] = ayb

                        encoded_label[i][j][k] = 0

                #print(axa,aya,axb,ayb)
    return localization,encoded_label



def encode_layers(batchsize,batch_labels, featuremap_widths,scales,ratios):
    batch_encodes_c=[]
    batch_encodes_r=[]

    for i,width in enumerate(featuremap_widths):
        encoded_local = []
        encoded_class = []
        anchors = gen_anchor_for_1_layer(width, scales[i], ratios)

        for labels in batch_labels:

            local, encoded_labels=encode_labels_1_layer(labels, anchors)

            encoded_local.append(local)
            encoded_class.append(encoded_labels)
       # batch_encodes[width]=[np.array(encoded_class),np.array(encoded_local)]
        batch_encodes_c.append(np.array(encoded_class))
        batch_encodes_r.append(np.array(encoded_local))

        #print(np.array(encoded_class).shape)
        #print("test", np.array(encoded_local).shape)

    return batch_encodes_c,batch_encodes_r


def fetch_data(batch_size,img_dir,ann_dir):

    batch_labels=[]
    batch_img=[]
    for i in range(0,batch_size):
        img, labels = read_frame("frame0.jpg", img_dir, ann_dir)
        img = cv2.resize(img, (512, 512))
        batch_labels.append(convert_labels_to_xyhw(labels))
        batch_img.append(img)
    return batch_img,batch_labels


#print(labels[0][1][2])


import cv2
import os
import sys
import xml.etree.ElementTree as ET
import glob
import math

def read_frame(filname, img_path="", ann_path=""):
    img_path = os.path.join(img_path, filname)
    ann_path = os.path.join(ann_path, filname[:-3] + "xml")

    img = cv2.imread(img_path)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # actual parsing
    in_file = open(ann_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    jpg = str(root.find('filename').text)
    imsize = root.find('size')
    w = int(imsize.find('width').text)
    h = int(imsize.find('height').text)
    label = list()
    bbox=list()
    dumps = list()


    for obj in root.iter('object'):
        current = list()
        name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))
        #label+=[[name]]
        label+=[[1]]

        bbox+=[[xn, yn, xx, yx]]

    in_file.close()
    # print(add)
    return img, w,h,label,bbox


def convert_labels_to_xyhw(bboxes, img_w, img_h):

    for i,label in enumerate(bboxes):
        xn, yn, xx, yx = label
        #print(label)
        #print(xn, yn, xx, yx)
        x = (xn + xx) / 2
        y = (yn + yx) / 2
        w = (xx - xn) / 2
        h = (yx - yn) / 2
        #print(labels[0][1][2][i])
        bboxes[i][0]= x / img_w*512
        bboxes[i][1]= y / img_h*512
        bboxes[i][2]= w / img_w*512
        bboxes[i][3]= h / img_h*512

        #print(labels[0][1][2][i][1], labels[0][1][2][i][2], labels[0][1][2][i][3], labels[0][1][2][i][4])
    #print("labels=",labels)
    return bboxes

def convert_to_placeholder_form(labelsnbboxes):
    return

# img,labels=read_frame("frame0.jpg")

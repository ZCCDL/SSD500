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
    all = list()
    dumps = list()

    for obj in root.iter('object'):
        current = list()
        name = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xn = int(float(xmlbox.find('xmin').text))
        xx = int(float(xmlbox.find('xmax').text))
        yn = int(float(xmlbox.find('ymin').text))
        yx = int(float(xmlbox.find('ymax').text))
        current = [name, xn, yn, xx, yx]
        all += [current]

    add = [[jpg, [w, h, all]]]
    in_file.close()
    # print(add)
    return img, add


def convert_labels_to_xyhw( labels):
    img_w, img_h = labels[0][1][0], labels[0][1][1]
    # print(labels)

    for i,label in enumerate(labels[0][1][2]):
        lb, xn, yn, xx, yx = label
        #print(label)
        #print(xn, yn, xx, yx)
        x = (xn + xx) / 2
        y = (yn + yx) / 2
        w = (xx - xn) / 2
        h = (yx - yn) / 2
        #print(labels[0][1][2][i])
        labels[0][1][2][i][1]=x/img_w
        labels[0][1][2][i][2]=y/img_h
        labels[0][1][2][i][3]=w/img_w
        labels[0][1][2][i][4]=h/img_h

        #print(labels[0][1][2][i][1], labels[0][1][2][i][2], labels[0][1][2][i][3], labels[0][1][2][i][4])
    #print("labels=",labels)
    return labels

# img,labels=read_frame("frame0.jpg")

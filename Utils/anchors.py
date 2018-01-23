import numpy as np
import math
import cv2
import random


def getscales(m):
    scales = []
    for k in range(1, m + 1):
        min = 0.04
        max = 0.90
        scales.append((min + (((max - min) / (m - 1)) * (k - 1))))
        print(scales)
    return scales


def generate_anchors(scales, aspectratio=[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], featuremapwidths=[64, 32, 16, 8, 4, 2, 1],
                     img=np.ones((512, 512, 3), dtype="uint8")):
    inp = 512

    print("scales", scales)
    for c in range(0, 6):
        scale = scales[c]
        # scale = scale * inp

        featuremapwidth = featuremapwidths[c]
        print(scale, featuremapwidth)
        for i in range(0, featuremapwidth):
            for j in range(0, featuremapwidth):
                color = (random.randint(0, 200), 150, random.randint(0, 150))

                cv2.line(img, (int((i + 1) * inp / featuremapwidth), 0), (int((i + 1) * inp / featuremapwidth), 512),
                         color=(255, 255, 255))
                cv2.line(img, (0, int((j + 1) * inp / featuremapwidth)), (512, int((j + 1) * inp / featuremapwidth)),
                         color=(255, 255, 255))

                for a in aspectratio:
                    w = math.sqrt(a) * scale
                    h = scale / math.sqrt(a)
                    print(w, h, "a=", a)
                    featwidth = inp / featuremapwidth

                    w = int(w * inp)
                    h = int(h * inp)
                    x = int((i + .5) * featwidth)
                    y = int((j + .5) * featwidth)

                    if i == 1 and j == 1:
                        cv2.circle(img,(x,y),0,color=(255, 255, 255),thickness=1)

                        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color=color)

        cv2.imshow("sadsa" + str(featuremapwidth), img)
        # cv2.waitKey(3000)

        img = np.ones((512, 512, 3), dtype="uint8")

    cv2.waitKey(0)

    return


def gen_anchor_for_1_layer(feature_map_size, scale,aspect_ratios,img_size=512 ):
    np.array((feature_map_size,feature_map_size,len(aspect_ratios),4))


    y,x=np.mgrid[0:feature_map_size, 0:feature_map_size]
    y=(y.astype(np.float32)+0.5)*(img_size/feature_map_size)/img_size
    x=(x.astype(np.float32)+0.5)*(img_size/feature_map_size)/img_size
    print(x,y)

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)



    num_anchors=len(aspect_ratios)

    w = np.zeros((num_anchors, ), dtype=np.float32)
    h = np.zeros((num_anchors, ), dtype=np.float32)




    for i,a in enumerate(aspect_ratios):
        w[i] = math.sqrt(a) * scale
        h[i] = scale / math.sqrt(a)
        print(w, h, "a=", a)
    return x,y,h,w


gen_anchor_for_1_layer(64,getscales(7)[5],[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0])

#generate_anchors(getscales(7))


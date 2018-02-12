import cv2
import random
from Utils.anchors import gen_anchor_for_1_layer
def visual(img, encodes, ious, aspectratio, scales):
    print(len(encodes))
    for img,mat in enumerate(img):
        for feat in range(0,7):
            print(encodes[feat].shape)
            for i in range(0, encodes[feat].shape[1]):
                for j in range(0, encodes[feat].shape[2]):
                    for k in range(0, encodes[feat].shape[3]):
                        x=encodes[feat][img][i][j][k][0]
                        y=encodes[feat][img][i][j][k][1]
                        w=encodes[feat][img][i][j][k][2]
                        h=encodes[feat][img][i][j][k][3]

                        if(ious[feat][img][i][j][k]>0):
                            print("-",i,j,x, y, w, h,ious[feat][img][i][j][k])
                        x=x*512
                        y=y*512
                        w=w*512
                        h=h*512

                        xmin=int(x-w)
                        ymin=int(y-h)

                        xmax=int(x+w)
                        ymax=int(y+h)

                        #print(xmin, xmax, xmax,ymax)

                        cv2.rectangle(mat,(xmin,ymin),(xmax,ymax),color=(255,255,255))
                        #cv2.putText(img,str(ious[feat][i][j][k]),(xmin,ymin) ,cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255))


        cv2.imshow("aa"+str(random.randint(0,232432434)),mat)
    cv2.waitKey(0)

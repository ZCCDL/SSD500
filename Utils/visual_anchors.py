import cv2
import random
def visual(img,anchors,ious,aspectratio,scales):
    print(len(anchors))
    for img,mat in enumerate(img):
        for feat in range(0,7):
            print(anchors[feat].shape)
            for i in range(0,anchors[feat].shape[1]):
                for j in range(0, anchors[feat].shape[2]):
                    for k in range(0, anchors[feat].shape[3]):
                        x=anchors[feat][img][i][j][k][0]
                        y=anchors[feat][img][i][j][k][1]
                        w=anchors[feat][img][i][j][k][2]
                        h=anchors[feat][img][i][j][k][3]

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

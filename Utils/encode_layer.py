import numpy as np
import math

from Utils.iou import bb_intersection_over_union


def encode_labels_1_layer(labels, anchors):
    ax = anchors[0]
    ay = anchors[1]
    aw = anchors[2]
    ah = anchors[3]
    #ax=np.expand_dims(ax,axis=-1)
    #ay=np.expand_dims(ay,axis=-1)
    bboxes = np.zeros((ax.shape[0], ax.shape[1], len(aw), 4), dtype=np.float32)
    ious = np.zeros((ax.shape[0], ax.shape[1], len(aw)), dtype=np.float32)
    en_labels = np.zeros((ax.shape[0], ax.shape[1], len(aw)), dtype=np.int64)
    print(ax.shape)
    dim4th=[]
    for i in range(0, len(aw)):
        width = ax.copy()
        height = ax.copy()
        print(width.shape)
        width[:, :] = aw[i]
        height[:, :] = ah[i]
        print(width[0][0])
        stacked = np.stack((ax, ay, width, height), axis=2)
        dim4th.append(stacked)

        print(stacked.shape)
        print(stacked[0][0])

    bboxes=np.stack((dim4th[0],dim4th[1]),axis=2)
    print(bboxes.shape)
    for i in range(2,len(dim4th)):
        dim4th[i]=np.expand_dims(dim4th[i],axis=2)
        print(dim4th[i].shape)
        bboxes=np.stack((bboxes,dim4th[i]))


    #print(ax.shape)
    #print(ay.shape)
    #stacked = np.stack((ax, ay), axis=2)

    # print(stacked.shape)
    # print(ax[0][6])
    # print(ay[0][6])
    # print(stacked[0][6])


    '''
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

                    if iou > ious[i][j][k] and iou > 0.5:
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
    '''

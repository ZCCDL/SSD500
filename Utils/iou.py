def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # print "max of" ,boxA[0], boxB[0] ,"is",xA
    # print "max of", boxA[1], boxB[1] ,"is",yA
    # print "min of", boxA[2], boxB[2] ,"is",xB
    # print "min of" ,boxA[3], boxB[3] ,"is",yB
    # compute the area of intersection rectangle

    dx = (xB - xA + 0.000000001)
    dy = (yB - yA + 0.000000001)
    if (dx >= 0) and (dy >= 0):
        interArea = dx * dy
    else:
        interArea = 0
    # compute the area of both the prediction and ground-truth
    # rectangles

    boxAArea = (boxA[2] - boxA[0] + 0.0000001) * (boxA[3] - boxA[1] + 0.0000001)
    boxBArea = (boxB[2] - boxB[0] + 0.0000001) * (boxB[3] - boxB[1] + 0.0000001)
    totalArea = float(boxAArea + boxBArea - interArea)

    iou = interArea / totalArea

    return iou

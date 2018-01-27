import tensorflow as tf


def l1_loss(gt, logits, thresh=1):
    x = tf.abs(tf.subtract(gt, logits))

    x = tf.where(x < thresh, 0.5 * x ** 2, thresh * (x - 0.5 * thresh))
    return x


def tloss(groundtruth_c, groundtruth_r,logits_c,logits_r):


    total_cls_loss = 0.0
    total_reg_loss = 0.0
    for i in range(0, len(groundtruth_c)):
        cls = logits_c[i]
        reg = logits_r[i]
        gcls = groundtruth_c[i]
        greg = groundtruth_r[i]


        #gcls=tf.reshape(gcls,gcls.shape[:-)
        print(cls.shape)
        print(gcls.shape)
        # print(gcls)
        # todo smooth l1 tloss
        pmask = gcls > 0
        fpmask = tf.cast(pmask, tf.float64)
        #fpmask=tf.reshape(fpmask,fpmask.get_shape().as_list()[:-1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gcls, logits=cls)
        loss = tf.losses.compute_weighted_loss(loss, fpmask)
        loss=tf.reduce_mean(loss)
        total_cls_loss += loss

        reg_loss = l1_loss(greg, reg)
        reg_loss = tf.losses.compute_weighted_loss(reg_loss, tf.expand_dims(fpmask,-1))
        reg_loss=tf.reduce_mean(reg_loss)

        total_reg_loss+=reg_loss


    return (total_reg_loss+total_cls_loss)

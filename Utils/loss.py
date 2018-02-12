import tensorflow as tf
import numpy


def l1_loss(gt, logits, thresh=1):
    with tf.variable_scope("L1_Loss"):
        absolute_loss = tf.abs(gt - logits)
        square_loss = 0.5 * (gt - logits) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)

    return tf.reduce_sum(l1_loss, axis=-1)




def tloss(groundtruth_c, groundtruth_r, logits_c, logits_r):
    total_class_loss = []
    total_reg_loss = []
    n_pos = 0

    for i in range(0, len(groundtruth_c)):
    #for i in range(0, 1):
        pred_c = logits_c[i]
        pred_r = logits_r[i]
        true_c = groundtruth_c[i]
        true_r = groundtruth_r[i]

        print(true_c)
        print(true_c.get_shape(), pred_c.get_shape())
        with tf.variable_scope("total-loss"):
            with tf.variable_scope("pos-mask"):

                '''Positive mask'''
                pmask_cond = true_c > 0
                pos_mask = tf.cast(pmask_cond, tf.float32)
                n_pos += tf.count_nonzero(true_c)
            with tf.variable_scope("neg-mask"):

                '''Negative mask'''
                nmask_cond = true_c == 0
                neg_mask = tf.cast(nmask_cond, tf.float32)
                classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_c, labels=true_c)

            with tf.variable_scope("cls-score"):
                '''Positive anchors loss'''
                positive_class_loss = classification_loss * pos_mask
                total_positive_class_loss = tf.reduce_sum(positive_class_loss)

                '''Negative  mining'''
                negative_class_loss = classification_loss * neg_mask
                n_neg_class_loss = tf.count_nonzero(negative_class_loss, dtype=tf.int32)
                n_negative_keep = tf.minimum(3 * tf.to_int32(n_pos), n_neg_class_loss)  # 3:1 mining

                '''Reshape find max loss indexes and change back to origial shape'''
                shape = negative_class_loss.get_shape()
                flat_neg_losses = tf.reshape(negative_class_loss, tf.TensorShape([shape[0] * shape[1] * shape[2] * shape[3]]))
                values, indices = tf.nn.top_k(flat_neg_losses, n_negative_keep)
                negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                               updates=tf.ones_like(indices, dtype=tf.int32),
                                               shape=tf.shape(flat_neg_losses))
                negatives_keep = tf.to_float(
                    tf.reshape(negatives_keep, tf.TensorShape([shape[0], shape[1], shape[2], shape[3]])))

                '''Neg Loss'''
                neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep)  # Tensor of shape (batch_size,)

                '''Total Classification Loss'''
                total_class_loss.append(total_positive_class_loss + neg_class_loss)

            '''Regression Loss'''
            with tf.variable_scope("reg-score"):

                reg_loss = l1_loss(true_r, pred_r) * pos_mask
                total_reg_loss.append(tf.reduce_sum(reg_loss))

    with tf.variable_scope("LOSSSSSS"):

        total_loss=0
        for i,reg_l in enumerate(total_reg_loss):
            total_loss+=total_reg_loss[i]+total_class_loss[i]

        total_loss=total_loss/ tf.cast(n_pos, dtype=tf.float32)
        tf.summary.scalar("loss",total_loss)


    return total_loss

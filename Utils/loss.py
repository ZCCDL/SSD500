import tensorflow as tf
import numpy


def l1_loss(gt, logits, thresh=1):
    absolute_loss = tf.abs(gt - logits)
    square_loss = 0.5 * (gt - logits) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def log_loss(y_pred, y_true):
    '''
    Compute the softmax log loss.
    Arguments:
        y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
            In this context, the expected tensor has shape (batch_size, #boxes, #classes)
            and contains the ground truth bounding box categories.
        y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
            the predicted data, in this context the predicted bounding box categories.
    Returns:
        The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
        of shape (batch, n_boxes_total).
    '''
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)

    return log_loss


def tloss(groundtruth_c, groundtruth_r, logits_c, logits_r):
    total_class_loss = 0.0
    total_reg_loss = 0.0
    n_pos = 0

    for i in range(0, len(groundtruth_c)):
        pred_c = logits_c[i]
        pred_r = logits_r[i]
        true_c = groundtruth_c[i]
        true_r = groundtruth_r[i]

        print(true_c.get_shape(), pred_c.get_shape())

        '''Positive mask'''
        pmask_cond = true_c > 0
        pos_mask = tf.cast(pmask_cond, tf.float32)
        n_pos += tf.count_nonzero(true_c)

        '''Negative mask'''
        nmask_cond = true_c == 0
        neg_mask = tf.cast(nmask_cond, tf.float32)

        conf = tf.nn.softmax(logits=pred_c)
        classification_loss = log_loss(conf, tf.one_hot(true_c, depth=conf.get_shape()[4], dtype=tf.float32))

        '''Positive anchors loss'''
        positive_class_loss = classification_loss * pos_mask
        total_positive_class_loss = tf.reduce_sum(positive_class_loss)

        '''Negative  mining'''
        negative_class_loss = classification_loss * neg_mask
        n_neg_class_loss = tf.count_nonzero(negative_class_loss, dtype=tf.int32)
        n_negative_keep = tf.minimum(3 * tf.to_int32(n_pos), n_neg_class_loss)  # 3:1 mining

        '''Reshape find max loss indexes and change back to origial shape'''
        shape = negative_class_loss.get_shape()
        flat_neg_losses = tf.reshape(negative_class_loss, [shape[0] * shape[1] * shape[2] * shape[3]])
        values, indices = tf.nn.top_k(flat_neg_losses, n_negative_keep)
        negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(flat_neg_losses))
        negatives_keep = tf.to_float(tf.reshape(negatives_keep, [shape[0], shape[1], shape[2], shape[3]]))

        '''Neg Loss'''
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep)  # Tensor of shape (batch_size,)

        '''Total Classification Loss'''
        total_class_loss += total_positive_class_loss + neg_class_loss

        '''Regression Loss'''
        reg_loss = l1_loss(true_r, pred_r) * pos_mask
        total_reg_loss += tf.reduce_sum(reg_loss)

        # total_reg_loss+=reg_loss

    return (total_reg_loss + total_class_loss) / tf.cast(n_pos, dtype=tf.float32)
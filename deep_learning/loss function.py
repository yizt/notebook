# -*- coding: utf-8 -*-
"""
   File Name：     loss function
   Description :  损失函数总结
   Author :       mick.yi
   date：          2019/1/29
"""
import keras.backend as K
import tensorflow as tf
import keras


def smooth_l1_loss(y_true, y_predict):
    """
    smooth L1损失函数；   0.5*x^2 if |x| <1 else |x|-0.5; x是 diff
    :param y_true:[N,4]
    :param y_predict:[N,4]
    :return:
    """
    abs_diff = tf.abs(y_true - y_predict, name='abs_diff')
    loss = tf.where(tf.less(abs_diff, 1), 0.5 * tf.pow(abs_diff, 2), abs_diff - 0.5)
    return tf.reduce_mean(loss, axis=1)


def softamx():
    y_true = tf.constant([[[0, 1, 0]]],
                         dtype=K.floatx())
    y_pred = tf.constant([[[0, 0.8, 0.2]]],
                         dtype=K.floatx())
    loss = K.categorical_crossentropy(y_true, y_pred)
    y_true = tf.constant([[[0, 1, 0, 0]]],
                         dtype=K.floatx())
    y_pred = tf.constant([[[0, 0.8, 0.1, 0.1]]],
                         dtype=K.floatx())
    loss2 = K.categorical_crossentropy(y_true, y_pred)
    sess = K.get_session()
    print(sess.run(loss))
    print(sess.run(loss2))


def main():
    y_true = tf.constant([[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]],
                         dtype=K.floatx())
    y_pred = tf.constant([[[0, 1, 0], [0, 2, 0], [1, 3, 1], [-1, 1, -1], [-2, 0, -2], [-1, 3, -1], [0, -1, 0]]],
                         dtype=K.floatx())
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_binary = K.binary_crossentropy(y_true, y_pred, from_logits=True)
    loss_weighted = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=1.0)
    # K.ctc_batch_cost()
    sess = K.get_session()
    print(sess.run(loss))
    print(sess.run(tf.reduce_mean(loss_sigmoid, axis=-1)))
    print(sess.run(loss_sigmoid))
    print(sess.run(loss_binary))
    print(sess.run(loss_weighted))


def test():
    x = tf.ones(shape=[3, 5, 4])
    y = tf.reduce_sum(x[..., ::2], axis=-1)
    print(y)


if __name__ == '__main__':
    # main()
    # softamx()
    test()
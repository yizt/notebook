# -*- coding: utf-8 -*-
"""
   File Name：     loss function
   Description :  损失函数总结
   Author :       mick.yi
   date：          2019/1/29
"""
import keras.backend as K
import tensorflow as tf

if __name__ == '__main__':
    y_true = tf.constant([[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]],
                         dtype=K.floatx())
    y_pred = tf.constant([[[0, 1, 0], [0, 2, 0], [1, 3, 1], [-1, 1, -1], [-2, 0, -2], [-1, 3, -1], [0, -1, 0]]],
                         dtype=K.floatx())
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss_sigmoid = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss_binary = K.binary_crossentropy(y_true, y_pred, from_logits=True)
    loss_weighted = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=1.0)
    K.ctc_batch_cost()
    sess = K.get_session()
    print(sess.run(loss))
    print(sess.run(tf.reduce_mean(loss_sigmoid, axis=-1)))
    print(sess.run(loss_sigmoid))
    print(sess.run(loss_binary))
    print(sess.run(loss_weighted))

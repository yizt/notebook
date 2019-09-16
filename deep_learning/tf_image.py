# -*- coding: utf-8 -*-
"""
   File Name：     tf_image
   Description :   tf.image 常用方法
   Author :       mick.yi
   date：          2019/2/1
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def test():
    img = plt.imread('../../data/1.jpg')
    shape = img.shape
    img = img.reshape([1, shape[0], shape[1], shape[2]])
    a = tf.image.crop_and_resize(img, [[0.5, 0.5, 0.6, 0.2], [0.5, 0.5, 1.3, 0.9]],
                                 box_ind=[0, 0], crop_size=(100, 100))
    sess = tf.Session()
    b = a.eval(session=sess)
    plt.subplot
    plt.imshow(b[0] / 255)
    plt.imshow(b[0].astype('uint8'))
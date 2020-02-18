# -*- coding: utf-8 -*-
"""
 @File    : svd.py
 @Time    : 2020/2/18 下午2:33
 @Author  : yizuotian
 @Description    :
"""

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("../data/2.jpg")
    h, w, c = img.shape
    img_temp = img.reshape(h, w * c)
    U, Sigma, VT = np.linalg.svd(img_temp)
    cv2.imshow('origin', img)

    print("img_temp:{},U:{},Sigma:{},VT:{}".format(img_temp.shape,
                                                   U.shape,
                                                   Sigma.shape,
                                                   VT.shape))
    print(Sigma)

    eigen_num = 60
    img_svd = (U[:, 0:eigen_num]).dot(np.diag(Sigma[0:eigen_num])).dot(VT[0:eigen_num, :])
    img_svd = img_svd.reshape(h, w, c)

    cv2.imshow('svd_' + str(eigen_num), img_svd.astype(np.uint8))

    eigen_num = 120
    img_svd = (U[:, 0:eigen_num]).dot(np.diag(Sigma[0:eigen_num])).dot(VT[0:eigen_num, :])
    img_svd = img_svd.reshape(h, w, c)
    cv2.imshow('svd_' + str(eigen_num), img_svd.astype(np.uint8))
    cv2.waitKey(0)

    np.

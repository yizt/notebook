# -*- coding: utf-8 -*-
"""
 @File    : pca.py
 @Time    : 2020/2/18 下午3:12
 @Author  : yizuotian
 @Description    :
"""
import numpy as np


def pca(data, k):
    n_samples, n_features = data.shape
    # 求均值
    mean = np.mean(data, axis=0)
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵
    matrix_ = np.dot(np.transpose(normal_data), normal_data)

    u, s, v = np.linalg.svd(matrix_)

    # new_normal_data = u[:n_samples, :k].dot(np.diag(s[:k])).dot(v[:k, :])

    new_normal_data = normal_data.dot(u[:, :k]).dot(u[:, :k].T)

    new_data = new_normal_data + mean

    return new_data


if __name__ == '__main__':
    x = np.abs(np.random.randn(4, 4))
    eig_val, eig_vec = np.linalg.eig(x.T.dot(x))
    print(eig_vec)
    u, s, v = np.linalg.svd(x.T.dot(x))
    print(s)
    print(u.dot(u.T))
    print(u[:, :3].dot(u[:, :3].T))
    print(u.T.dot(u))
    # print(u)
    # print(v)
    # print(u.dot(v))
    # print(eig_val)
    # print(s)
    # img = cv2.imread("../data/2.jpg")
    # h, w, c = img.shape
    # img_temp = img.reshape(h, w * c)
    # k = 60
    # img_pca = pca(img_temp, k)
    # print(img_pca.shape)
    # img_pca = img_pca.reshape(h, w, c)
    # cv2.imshow('origin', img)
    # cv2.imshow('pca_' + str(k), img_pca.astype(np.uint8))
    # cv2.waitKey(0)

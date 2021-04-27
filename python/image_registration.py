# -*- coding: utf-8 -*-
"""
 @File    : image_registration.py
 @Time    : 2021/4/27 上午8:40
 @Author  : yizuotian
 @Description    :
"""
import os

import cv2
import numpy as np


def find_correspondence_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T, retval


def perspective_transform(src_pts, matrix):
    """
    对坐标点透视变换
    :param src_pts: 源坐标点 numpy [N,(x,y)]
    :param matrix: 转换矩阵 numpy [3,3]
    :return dst_pts: 目标坐标点 numpy [N,(x,y)]
    """
    num_points = src_pts.shape[0]
    constant = np.ones(shape=(num_points, 1))
    src_pts = np.concatenate([src_pts, constant], axis=1)  # [N,(x,y,1)]
    dst_pts = np.dot(matrix, src_pts.T).T  # [N,(x,y,1)]
    dst_pts[:, 0] /= dst_pts[:, 2]
    dst_pts[:, 1] /= dst_pts[:, 2]
    dst_pts = dst_pts[:, :2]  # [N,(x,y)]
    return dst_pts.astype(np.int32)


def draw_points(im, pts):
    """

    :param im:
    :param pts: numpy [N,(x,y)]
    :return:
    """
    for x, y in pts:
        cv2.circle(im, (x, y), 3, (0, 0, 255), 3)


def deal_im(src_pts, im_template, im):
    # 查找匹配点，获取透视变换矩阵
    _, _, matrix = find_correspondence_points(im_template, im)
    dst_pts = perspective_transform(src_pts, matrix)

    h, w = im.shape[:2]
    im_warp = cv2.warpPerspective(im_template,
                                  M=matrix,
                                  dsize=(w, h))
    im_dst = (0.5 * im + 0.5 * im_warp).astype(np.uint8)

    # 画点
    draw_points(im_dst, dst_pts)

    if h < w:
        return np.concatenate([im, im_dst], axis=0)
    return np.concatenate([im, im_dst], axis=1)


def main():
    pts = [[58, 239],
           [162, 342],
           [264, 242],
           [364, 342],
           [464, 242],
           [560, 340],
           [658, 242],
           [756, 340],
           [62, 648],
           [164, 746],
           [268, 644],
           [368, 742],
           [468, 640],
           [568, 736],
           [664, 636],
           [762, 732]]
    pts = np.array(pts)
    template_im_path = '/Volumes/Elements/土方智能工厂/template.jpg'
    src_im_dir = '/Volumes/Elements/土方智能工厂/棋盘格'
    out_dir = '/Volumes/Elements/土方智能工厂/match_output'

    im_template = cv2.imread(template_im_path)
    for im_name in os.listdir(src_im_dir):
        print('deal {}'.format(im_name))
        im_path = os.path.join(src_im_dir, im_name)
        im = cv2.imread(im_path)

        out_path = os.path.join(out_dir, im_name)

        im_out = deal_im(pts, im_template, im)
        cv2.imwrite(out_path, im_out)

    # draw_points(im_template, pts)
    # cv2.imwrite('/Volumes/Elements/土方智能工厂/template_draw.jpg', im_template)


if __name__ == '__main__':
    main()

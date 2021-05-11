# -*- coding: utf-8 -*-
"""
 @File    : main_valve.py
 @Time    : 2021/1/18 上午11:31
 @Author  : yizuotian
 @Description    :
"""
import os

import cv2
import numpy as np


def get_binary(im_gray):
    """
    获取灰度图的二值图
    :param im_gray:
    :return:
    """
    size = 3
    gx = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=size)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=size)
    gy = cv2.convertScaleAbs(gy)

    g = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    # return (g>120).astype(np.uint8)*255
    return g


def color_mask(hsv, gray, color):
    # h:0-180,s:0-255,v:0-255
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if color == 'gray':
        return np.logical_and(np.logical_and(v <= 220, gray <= 150),
                              np.logical_and(s <= 43, v >= 46))
    elif color == 'white':
        return np.logical_and(s <= 30, v >= 221)
        # return np.logical_and(np.logical_and(s <= 30, v >= 221),
        #                       gray <= 150)  # np.logical_or(gray <= 150, gray >= 200)
    elif color == 'black':
        return v <= 46
    elif color == 'red':
        return np.logical_and(np.logical_or(h >= 156, h <= 10),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'orange':
        return np.logical_and(np.logical_and(h >= 11, h <= 25),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'yellow':
        return np.logical_and(np.logical_and(h >= 26, h <= 34),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'green':
        return np.logical_and(np.logical_and(h >= 35, h <= 77),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'cyan':
        return np.logical_and(np.logical_and(h >= 78, h <= 99),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'blue':
        return np.logical_and(np.logical_and(h >= 100, h <= 124),
                              np.logical_and(s >= 43, v >= 46))
    elif color == 'purple':
        return np.logical_and(np.logical_and(h >= 125, h <= 155),
                              np.logical_and(s >= 43, v >= 46))


def get_all_color_mask(im_bgr):
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    color_list = [  # 'white',
        'gray', 'black', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
    mask_dict = {}
    for color in color_list:
        mask_dict[color] = color_mask(hsv, gray, color)

    return mask_dict


def deal_one_image(im_path):
    im = cv2.imdecode(np.fromfile(im_path, np.uint8), cv2.IMREAD_COLOR)
    img = im.copy()
    # 颜色区域提取
    mask = np.stack(list(get_all_color_mask(im).values()), axis=0).astype(np.uint32)
    mask_all = np.sum(mask, axis=0)
    mask_all = (mask_all == 0).astype(np.uint8) * 255
    # mask_all = get_all_color_mask(im)['white'].astype(np.uint8) * 255
    # print(np.min(mask_all))
    # Image.fromarray(mask_all)
    # 开区间
    open = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel=np.ones((3, 3)), iterations=5)
    # Image.fromarray(open)

    # 取最大连通域
    # _,
    _, contours, hierarchy = cv2.findContours((open).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area_idx = -1
    max_area = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area_idx = i
            max_area = area

    new_mask = np.zeros_like(mask_all, np.uint8)
    new_mask = cv2.fillConvexPoly(new_mask, contours[max_area_idx], 255)

    # 画直线
    minLineLength = 100
    maxLineGap = 20
    lines = cv2.HoughLinesP(get_binary(new_mask), 1, np.pi / 180, 60, minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 最小外接矩形
    rect = cv2.minAreaRect(contours[max_area_idx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 1)

    # 拼接图像
    zero_im = np.zeros_like(img)
    out_im = np.concatenate([mask_all[:, :, np.newaxis] + zero_im,
                             new_mask[:, :, np.newaxis] + zero_im,
                             img],
                            axis=1)

    return out_im


def deal_dir(im_dir, out_dir):
    for im_name in os.listdir(im_dir):
        im_path = os.path.join(im_dir, im_name)
        out_im = deal_one_image(im_path)
        out_path = os.path.join(out_dir, im_name)
        # cv2.imwrite(out_path, out_im)
        cv2.imencode('.jpg', out_im)[1].tofile(out_path)


if __name__ == '__main__':
    # mask_all, new_mask, img = deal_one_image('main_valve/1.01.jpg')
    #
    # zero_im = np.zeros_like(img)
    # out_im = np.concatenate([mask_all[:, :, np.newaxis] + zero_im,
    #                          new_mask[:, :, np.newaxis] + zero_im,
    #                          img],
    #                         axis=1)
    # cv2.imshow('', out_im)
    # cv2.waitKeyEx(0)
    deal_dir('/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-crop/box',
             '/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-crop/direct_out')

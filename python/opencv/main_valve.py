# -*- coding: utf-8 -*-
"""
 @File    : main_valve.py
 @Time    : 2021/1/18 上午11:31
 @Author  : yizuotian
 @Description    :
"""
import math
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
        # return np.logical_and(np.logical_and(v <= 220, gray <= 150),
        #                       np.logical_and(s <= 43, v >= 46))
        return np.logical_and(np.logical_and(v <= 220, gray <= 50),
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
    print(im_path)
    im = cv2.imdecode(np.fromfile(im_path, np.uint8), cv2.IMREAD_COLOR)
    # im = cv2.resize(im, (250, 250))
    img = im.copy()
    # 颜色区域提取
    mask = np.stack(list(get_all_color_mask(im).values()), axis=0).astype(np.uint32)
    mask_all = np.sum(mask, axis=0)
    mask_all = (mask_all == 0).astype(np.uint8) * 255
    # mask_all = get_all_color_mask(im)['white'].astype(np.uint8) * 255
    # print(np.min(mask_all))
    # Image.fromarray(mask_all)
    # 开区间
    open = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel=np.ones((3, 3)), iterations=3)
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

    if max_area_idx == -1:
        return im
    new_mask = np.zeros_like(mask_all, np.uint8)
    new_mask = cv2.fillConvexPoly(new_mask, contours[max_area_idx], 255)

    # 直线检测
    minLineLength = 100
    maxLineGap = 20
    lines = cv2.HoughLinesP(get_binary(new_mask), 1, np.pi / 180, 60, minLineLength, maxLineGap)

    # 最小外接矩形
    rect = cv2.minAreaRect(contours[max_area_idx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print(box,im_path)
    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

    # 画最终的角度
    if lines is not None and len(lines) > 0:
        angle, valid_flags = get_angle(box, lines)
        dx = np.cos(angle * np.pi / 180)
        dy = np.sin(angle * np.pi / 180)
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, h // 2), (int(w / 2 + dx * w / 6), int(h / 2 + dy * h / 6)), (255, 0, 0), 2)

        # 画直线
        for line, valid in zip(lines, valid_flags):
            x1, y1, x2, y2 = line[0]
            if valid:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 拼接图像
    zero_im = np.zeros_like(img)
    out_im = np.concatenate([mask_all[:, :, np.newaxis] + zero_im,
                             new_mask[:, :, np.newaxis] + zero_im,
                             img],
                            axis=1)

    return out_im


def deal_one_image_show(im_path):
    """
    展示
    :param im_path:
    :return:
    """
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

    # 直线检测
    minLineLength = 100
    maxLineGap = 20
    lines = cv2.HoughLinesP(get_binary(new_mask), 1, np.pi / 180, 60, minLineLength, maxLineGap)

    # 最小外接矩形
    rect = cv2.minAreaRect(contours[max_area_idx])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print(box,im_path)
    img_box = img.copy()
    cv2.drawContours(img_box, [box], 0, (0, 255, 255), 2)

    # 画最终的角度
    angle, valid_flags = get_angle(box, lines)
    dx = np.cos(angle * np.pi / 180)
    dy = np.sin(angle * np.pi / 180)
    h, w = img.shape[:2]
    img_angle = img.copy()
    cv2.line(img_angle, (w // 2, h // 2), (int(w / 2 + dx * w / 6), int(h / 2 + dy * h / 6)), (255, 0, 0), 2)

    # 画直线
    img_lines = img_box.copy()
    for line, valid in zip(lines, valid_flags):
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_valid_lines = img_box.copy()
    for line, valid in zip(lines, valid_flags):
        x1, y1, x2, y2 = line[0]
        if valid:
            cv2.line(img_valid_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 拼接图像
    # zero_im = np.zeros_like(img)
    # out_im = np.concatenate([mask_all[:, :, np.newaxis] + zero_im,
    #                          new_mask[:, :, np.newaxis] + zero_im,
    #                          img],
    #                         axis=1)
    # cv2.imshow('img', img)
    # cv2.imshow('mask_all', mask_all)
    # cv2.imshow('new_mask', new_mask)
    # cv2.imshow('img_box', img_box)
    cv2.imshow('img_lines', img_lines)
    # cv2.imshow('img_valid_lines', img_valid_lines)
    # cv2.imshow('img_angle', img_angle)
    # cv2.imshow('open', open)
    cv2.waitKeyEx(0)
    # return out_im


def get_angle(rect_pts, lines):
    """
    根据
    :param rect_pts: 矩形框顶点坐标[4,(x,y)]
    :param lines: 检测的直线[N,(x,y)]
    :return:
    """

    def _get_angle(p1_y, p1_x, p2_y, p2_x):
        if p2_y - p1_y > 0:
            return math.atan2(p2_y - p1_y, p2_x - p1_x) * 180 / math.pi
        return math.atan2(p1_y - p2_y, p1_x - p2_x) * 180 / math.pi

    # 矩形坐标
    p1, p2, p3, p4 = rect_pts
    angles = [_get_angle(p1[1], p1[0], p2[1], p2[0]),
              _get_angle(p3[1], p3[0], p2[1], p2[0])]
    angles = [angle if angle >= 0 else angle + 180 for angle in angles]
    a1, a2 = angles
    # rect_angle = min(a1, a2)
    # print(a1, a2)
    l1 = np.sqrt(np.square(p1[1] - p2[1]) + np.square(p1[0] - p2[0]))
    l2 = np.sqrt(np.square(p3[1] - p2[1]) + np.square(p3[0] - p2[0]))
    rect_angle = a1 if l1 >= l2 else a2
    rect_normal_line_angle = a2 if l1 >= l2 else a1  # 法线角度

    # 直线的角度
    lines_angle = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = _get_angle(y1, x1, y2, x2)
        lines_angle.append(angle)

    # 直线与矩形框的角度差
    deltas = [min(np.abs(angle - a1), np.abs(angle - a2)) for angle in lines_angle]
    # 过滤掉过大的角度差
    mean_delta = np.mean(deltas)
    valid_lines_angle = [angle for delta, angle in zip(deltas, lines_angle) if delta < mean_delta and delta < 15]
    valid_flags = [delta < mean_delta for delta, angle in zip(deltas, lines_angle)]

    # 有效直线的平均角度
    delta_sum = 0
    for angle in valid_lines_angle:
        if np.abs(angle - rect_angle) <= np.abs(angle - rect_normal_line_angle):
            delta_sum += (angle - rect_angle)
        else:
            delta_sum += (angle - rect_normal_line_angle)

    if len(valid_lines_angle) > 0:
        print(delta_sum / len(valid_lines_angle))
        return rect_angle + delta_sum / len(valid_lines_angle), valid_flags
    return rect_angle, valid_flags


def azimuth_angle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return angle * 180 / math.pi


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
    # deal_dir('/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-办公室-crop/box',
    #          '/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-办公室-crop/direct_out')
    deal_dir('/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-相机-crop/box',
             '/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-相机-crop/direct_out')

    # deal_one_image_show('/Volumes/Elements/土方智能工厂/工步防错-主阀/接头朝向-crop/box/IMG_20210506_104144_1_001.jpg')

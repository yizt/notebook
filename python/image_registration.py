# -*- coding: utf-8 -*-
"""
 @File    : image_registration.py
 @Time    : 2021/4/27 上午8:40
 @Author  : yizuotian
 @Description    :
"""
import colorsys
import json
import os

import cv2
import numpy as np


def find_correspondence_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.AKAZE_create()
    # sift = cv2.xfeatures2d.SURF_create(400)
    # sift = cv2.ORB_create()

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

    # flann = cv2.BFMatcher() # orb匹配
    matches = flann.knnMatch(des1,
                             des2, k=2)

    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # Apply Lowe's SIFT matching ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                      ransacReprojThreshold=20
                                      )
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
    """

    :param src_pts:
    :param im_template:
    :param im:
    :return:
    """
    # 查找匹配点，获取透视变换矩阵
    _, _, matrix = find_correspondence_points(im_template, im)
    # print(matrix)
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


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv]
    # random.shuffle(colors)
    colors = np.array(colors) * 255
    return colors


def draw_boxes(im, pts, colors):
    """

    :param im: [H,W,3]
    :param pts: [N,4,(x,y)]
    :param colors: [N,(B,G,R)]
    :return:
    """
    im_mask = np.zeros_like(im)

    for i, points in enumerate(pts):
        color = colors[i]
        cv2.fillConvexPoly(im_mask, points, color=color)

    im_new = (im_mask * 0.5 + im * 0.5).astype(np.uint8)

    mask = np.sum(im_mask, axis=-1)

    return np.where(mask[:, :, np.newaxis] > 0, im_new, im)


def parse_file(json_file):
    """
    解析vott标准
    :param json_file:
    :return:
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        regions = list(ret_dic['assets'].values())[0]['regions']
    pts = np.zeros(shape=(len(regions), 4, 2))  # [N,(lt,lr,br,bl),(x,y)]
    for i, region in enumerate(regions):
        for j, pt in enumerate(region['points']):
            pts[i, j, 0] = pt['x']
            pts[i, j, 1] = pt['y']

    return pts


def deal_im_valve(src_pts, im_template, im, colors):
    """

    :param src_pts: [N,4,2]
    :param im_template: [H,W,3]
    :param im: [H,W,3]
    :param colors: [N,(B,G,R)]
    :return:
    """
    # 查找匹配点，获取透视变换矩阵
    _, _, matrix = find_correspondence_points(im_template, im)
    # print(matrix)
    dst_pts = perspective_transform(src_pts.reshape(-1, 2), matrix)
    dst_pts = dst_pts.reshape(-1, 4, 2)
    im_draw = draw_boxes(im, dst_pts, colors)
    return im_draw


def valve_main():
    template_im_path = '/Volumes/Elements/土方智能工厂/工步防错-主阀/template/02.jpg'
    # src_im_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/位置校准'
    # out_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/位置校准_output_orb'

    src_im_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/ggg/1200_20-60'
    out_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/rgt_out/1200_20-60'
    json_path = '/Volumes/Elements/土方智能工厂/工步防错-主阀/template/main_valve2.json'

    pts = parse_file(json_path)
    colors = random_colors(len(pts))
    im_template = cv2.imread(template_im_path)
    for im_name in os.listdir(src_im_dir):
        print('deal {}'.format(im_name))
        im_path = os.path.join(src_im_dir, im_name)
        im = cv2.imread(im_path)

        out_path = os.path.join(out_dir, im_name)

        im_out = deal_im_valve(pts, im_template, im, colors)
        cv2.imwrite(out_path, im_out)

    # 模板图像展示
    img = draw_boxes(im_template,
                     pts.reshape(-1, 4, 2).astype(np.int32),
                     colors)
    out_path = os.path.join(out_dir, 'template.jpg')
    cv2.imwrite(out_path, img)


def valve_main_2():
    template_im_path = '/Volumes/Elements/土方智能工厂/工步防错-主阀/template.jpg'
    src_im_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/位置校准'
    out_dir = '/Volumes/Elements/土方智能工厂/工步防错-主阀/位置校准_output_2'
    json_path = '/Volumes/Elements/土方智能工厂/工步防错-主阀/main_valve.json'

    pts = parse_file(json_path).reshape(-1, 2)

    im_template = cv2.imread(template_im_path)
    for im_name in os.listdir(src_im_dir):
        print('deal {}'.format(im_name))
        im_path = os.path.join(src_im_dir, im_name)
        im = cv2.imread(im_path)

        out_path = os.path.join(out_dir, im_name)

        im_out = deal_im(pts, im_template, im)
        cv2.imwrite(out_path, im_out)


def chess_main():
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
    valve_main()

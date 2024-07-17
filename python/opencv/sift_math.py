# -*- coding: utf-8 -*-
"""
 @File    : sift_math.py
 @Time    : 2020/3/19 上午10:10
 @Author  : yizuotian
 @Description    :
"""
import argparse
import math

import cv2
import numpy as np


def sift_math(img1, img2, top_n=10):
    # sift = cv2.xfeatures2d.SIFT_create()

    sift = cv2.xfeatures2d.SURF_create(400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    # 比值测试，首先获取与 A距离最近的点 B （最近）和 C （次近），
    # 只有当 B/C 小于阀值时（0.75）才被认为是匹配，
    # 因为假设匹配是一一对应的，真正的匹配的理想距离为0
    good = []
    match_pt = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            match_pt.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    return good[:top_n], match_pt[:top_n]


def flann_match(img1, img2, top_n=10):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # # Need to draw only good matches, so create a mask
    # matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    match_pt = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            match_pt.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    return good[:top_n], match_pt[:top_n]


def get_angle(match_pt_list):
    angle_list = []
    for p1, p2 in match_pt_list:
        x1, y1 = p1
        x2, y2 = p2
        angle_list.append(math.atan2(y2 - y1, x2 - x1))

    angle_list.sort()
    angle = np.mean(angle_list[1:-1])
    return angle


def gen_arrowed_line(angle, scale=30):
    x0, y0 = scale // 2, scale // 2
    y = math.sin(angle) * y0 * 0.8
    x = math.cos(angle) * x0 * 0.8

    img = np.zeros((scale, scale, 3))

    img = cv2.arrowedLine(img, (x0, y0), (x0 + int(x), y0 + int(y)), (255, 0, 0), 1)
    return img


def main(args):
    cap = cv2.VideoCapture(args.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    size = int(np.sqrt(h * w) / 5)
    out = cv2.VideoWriter(args.output_video, fourcc, 20.0, (w, h))
    s_h, s_w = h // 5, w // 5
    flag = True
    frames = []
    while flag and cap.isOpened():
        flag, origin_frame = cap.read()

        if not flag:
            continue
        frame = cv2.resize(origin_frame, (s_w, s_h))
        frames.append(frame)
        if len(frames) > 5:
            frames = frames[1:]
        if len(frames) == 5:
            cur_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pre_img = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            _, match_pt_list = sift_math(cur_img, pre_img)
            # _, match_pt_list = flann_match(cur_img, pre_img)

            angle = get_angle(match_pt_list)

            img = gen_arrowed_line(angle, size)
            origin_frame[:size, :size] = img

        # cv2.imshow('origin', origin_frame)
        out.write(origin_frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-video', type=str,
                        default='/Users/yizuotian/demo_material/video_20200318_102620.mp4')
    parser.add_argument('-o', '--output-video', type=str, default='output.mp4')
    arguments = parser.parse_args()
    import time

    start = time.time()
    main(arguments)
    end = time.time()
    print(end - start)

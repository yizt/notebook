# -*- coding: utf-8 -*-
"""
 @File    : red_detector.py
 @Time    : 2020/3/18 上午11:01
 @Author  : yizuotian
 @Description    :
"""
import argparse

import cv2
import numpy as np


def main(args):
    cap = cv2.VideoCapture(args.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv 3.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(args.output_video, fourcc, 20.0, ((w // 3) * 2, h // 3))
    flag = True
    while flag and cap.isOpened():
        flag, frame1 = cap.read()
        if not flag:
            continue
        frame1 = cv2.resize(frame1, (w // 3, h // 3))
        idx_1 = np.logical_and(frame1[:, :, 2] > frame1[:, :, 0] + 20, frame1[:, :, 2] > frame1[:, :, 1] + 20)
        idx_2 = np.logical_and(frame1[:, :, 2] > frame1[:, :, 0] * 1.5, frame1[:, :, 2] > frame1[:, :, 1] * 1.5)
        idx = np.logical_and(np.logical_and(idx_1, idx_2), frame1[:, :, 2] > 150)

        img = np.ones_like(frame1) * 100

        # 膨胀看得更清楚
        kernel = np.ones((5, 5), np.uint8)
        idx = cv2.dilate(idx.astype(np.uint8), kernel, iterations=1)
        idx = idx.astype(np.bool)

        img[idx] = np.array([0, 0, 255])
        result = np.concatenate([frame1, img], axis=1)
        cv2.imshow('origin', result)
        out.write(result)
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
    main(arguments)

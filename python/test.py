# -*- coding: utf-8 -*-
"""
   File Name：     test.py
   Description :  
   Author :       mick.yi
   Date：          2019/6/14
"""

import os
import torch
import cv2


def video2images(file_path, output_path):
    cap = cv2.VideoCapture(file_path)
    flag = True
    i = 0
    while flag and cap.isOpened():
        flag, frame = cap.read()
        if frame is None:
            continue
        image_path = os.path.join(output_path, '{:03d}.jpg'.format(i))
        frame = cv2.resize(frame, (100, 100))
        cv2.imwrite(image_path, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def format_np(np_2d):
    '{}'.format([list(i) for i in np_2d])





if __name__ == '__main__':
    # video2images('/Users/yizuotian/zoomlion/demo_material/media2.mp4',
    #              '/Users/yizuotian/zoomlion/demo_material/pics/')
    # im = cv2.imread('√.png')
    # print(im.shape)
    # im = cv2.resize(im, dsize=(100, 100))
    # cv2.imwrite('gou.jpg', im)
    # torch.randn([2, 3])
    # torch.flatten()
    # torch.transpose()
    # torch.split()
    #
    # import torch.optim.sgd
    from clearml import Task

    task = Task.init(project_name="my project", task_name="my task")





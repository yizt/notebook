# -*- coding: utf-8 -*-
"""
   File Name：     video_utils.py
   Description :   视频处理工具类
   Author :       mick.yi
   Date：          2019/6/19
"""
import os

import cv2
from bokeh.io import output_notebook, show, push_notebook
from bokeh.plotting import figure
import numpy as np

def cv_show(file_path):
    cap = cv2.VideoCapture(file_path)
    flag = True
    while flag and cap.isOpened():
        flag, frame = cap.read()
        if frame is None:
            continue
        cv2.imshow('image', frame)
        k = cv2.waitKey(20)
        # q键退出
        if k & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def video2images(file_path, output_path):
    cap = cv2.VideoCapture(file_path)
    flag = True
    i = 0
    while flag and cap.isOpened():
        flag, frame = cap.read()
        if frame is None:
            continue
        image_path = os.path.join(output_path, '{:03d}.jpg'.format(i))
        cv2.imwrite(image_path, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def video_concat(video_path_list,out_path,axis=1):
    """
    拼接视频，假定视频宽高一致
    :param video_path_list:
    :param out_path:
    :param axis:
    :return:
    """
    caps = [cv2.VideoCapture(path) for path in video_path_list ]
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter()
    writer.open(out_path, fourcc, 25, (w*len(caps), h), True)
    flag = True
    i = 0
    while flag and np.alltrue([cap.isOpened() for cap in caps]):
        frames = []
        for cap in caps:
            cur_flag, frame = cap.read()
            if frame is None:
                continue
            flag = flag and cur_flag
            frames.append(frame)
        writer.write(np.concatenate(frames, axis=axis))
    for cap in caps:
        cap.release()
    writer.release()


class VideoShowInNotebook(object):
    def __init__(self, first_frame):
        """
        :param first_frame:
        """
        output_notebook()
        frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGBA)  # because Bokeh expects a RGBA image
        frame = cv2.flip(frame, 0)  # because Bokeh flips vertically
        width = frame.shape[1]
        height = frame.shape[0]
        p = figure(x_range=(0, width), y_range=(0, height), output_backend="webgl", width=width, height=height)
        self.image = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
        show(p, notebook_handle=True)

    def show(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame = cv2.flip(frame, 0)
        self.image.data_source.data['image'] = [frame]
        push_notebook()


def test_video_show():
    file_path = r'D:\pyspace\pytorch-video-recognition\data\Abuse011_x264.mp4'
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    vs = VideoShowInNotebook(frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            vs.show(frame)


def test_camera_show():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    vs = VideoShowInNotebook(frame)
    while cap.isOpened():
        ret, frame = cap.read()
        vs.show(frame)


if __name__ == '__main__':
    # cv_show('./data/Abuse011_x264.mp4')
    # cv_show(r'D:\pyspace\py_data_mining\anomaly-detection\tmp\predict-Abuse005_x264.mp4')
    # video2images('/Users/yizuotian/zoomlion/demo_material/video_20200318_102620.mp4',
    #              '/Users/yizuotian/zoomlion/demo_material/video_20200318_102620_pic')
    video_concat(['/Users/yizuotian/cspace/mix_station_embeded/output.mp4',
                  '/Users/yizuotian/cspace/mix_station_embeded/output-2.mp4'],
                 '/Users/yizuotian/cspace/mix_station_embeded/output-compare.mp4')

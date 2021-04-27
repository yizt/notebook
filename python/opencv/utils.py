# -*- coding: utf-8 -*-
"""
 @File    : utils.py
 @Time    : 2021/4/27 下午1:41
 @Author  : yizuotian
 @Description    :
"""
import os
import shutil

import cv2


def down_scale_image(im, im_size=1000):
    """
    缩小图像
    :param im:
    :param im_size:
    :return:
    """

    def _get_height_and_width(h, w, im_size):
        if h * w < im_size ** 2:
            return int(h), int(w)
        return _get_height_and_width(h * 0.9, w * 0.9, im_size)

    src_h, src_w = im.shape[:2]
    if src_h * src_w > im_size ** 2:
        dst_h, dst_w = _get_height_and_width(src_h, src_w, im_size)
        return cv2.resize(im, (dst_w, dst_h))
    return im


def down_scale_image_dir(dir_path, im_size):
    for im_name in os.listdir(dir_path):
        print('deal {}'.format(im_name))
        im_path = os.path.join(dir_path, im_name)
        im = cv2.imread(im_path)
        new_im = down_scale_image(im, im_size)
        cv2.imwrite(im_path, new_im)


def rename_file(dir_path, length=3):
    for i, im_name in enumerate(os.listdir(dir_path)):
        print('deal {}'.format(im_name))
        suffix = os.path.splitext(im_name)[1]
        dst_name = '{:09d}'.format(i + 1)[-length:]
        dst_name = '{}{}'.format(dst_name, suffix)
        src_path = os.path.join(dir_path, im_name)
        dst_path = os.path.join(dir_path, dst_name)
        shutil.move(src_path, dst_path)


def main():
    dir_path = '/Volumes/Elements/土方智能工厂/棋盘格/'
    down_scale_image_dir(dir_path, 800)
    rename_file(dir_path)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
 @File    : utils.py
 @Time    : 2021/4/27 下午1:41
 @Author  : yizuotian
 @Description    :
"""
import cv2


def down_scale_image(im, im_size=1000):
    """

    :param im:
    :param im_size:
    :return:
    """
    def _get_height_and_width(h, w, im_size):
        if h * w < im_size ** 2:
            return int(h), int(w)
        return _get_height_and_width(h * 0.9, w * 0.9, im_size)

    src_h, src_w = im.shape[:2]
    dst_h, dst_w = _get_height_and_width(src_h, src_w, im_size)
    return cv2.resize(im, (dst_w, dst_h))

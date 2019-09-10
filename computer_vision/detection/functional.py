# -*- coding: utf-8 -*-
"""
   File Name：     functional.py
   Description :  
   Author :       mick.yi
   Date：          2019/9/10
"""
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image
import sys
import collections
import numpy as np
import math

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.

    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img, h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

    else:
        ow, oh = size[::-1]
    return img.resize((ow, oh), interpolation), oh, ow


def rotate_boxes(boxes, angle, center):
    """
    x1 = cos(theta)x0 - sin(theta)y0
    y1 = sin(theta)x0 + cos(theta)y0
    :param boxes:
    :param angle:
    :param center:
    :return:
    """
    angle = angle * math.pi / 180
    ctr_y, ctr_x = center
    boxes[:, ::2] -= ctr_y
    boxes[:, 1::2] -= ctr_x

    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    p1 = np.concatenate([x1, y1], axis=1)
    p2 = np.concatenate([x2, y2], axis=1)
    # rotate
    delta = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])

    p1 = np.matmul(delta, p1.T).T
    p2 = np.matmul(delta, p2.T).T
    y1, y1 = np.split(p1, 2, axis=1)
    x2, y2 = np.split(p2, 2, axis=1)
    boxes = np.concatenate([y1, x1, y2, x2], axis=1)

    #
    boxes[:, ::2] += ctr_y
    boxes[:, 1::2] += ctr_x
    return boxes

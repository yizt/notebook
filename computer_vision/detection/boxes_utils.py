# -*- coding: utf-8 -*-
"""
   File Name：     boxes_utils.py
   Description :  
   Author :       mick.yi
   Date：          2019/9/10
"""
import colorsys
import random
import cv2
import numpy as np
from PIL import ImageDraw, Image


def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return colors


def draw_boxes(image, boxes, class_names=None, color='red'):
    draw_obj = ImageDraw.Draw(image)
    draw_obj.ink = sum([i * j for i, j in zip(random_color(), [1, 256, 256 * 256])])
    for i in range(len(boxes)):
        y1, x1, y2, x2 = boxes[i]
        draw_obj.rectangle((x1, y1, x2, y2), outline=color)
        if class_names is not None:
            draw_obj.text((x1, y1 - 10), class_names[i])
    return image


def draw_rect(image, boxes, class_names=None, color=None):
    """

    :param image:
    :param boxes:
    :param class_names:
    :param color:
    :return:
    """
    im = image.copy()
    boxes = boxes[:, :4].astype(np.int)
    if not color:
        color = random_color()
    for i in range(len(boxes)):
        y1, x1, y2, x2 = boxes[i]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color, int(max(im.shape[:2]) / 200))
        if class_names is not None:
            im = cv2.addText(im, class_names[i], (x1, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (255, 255, 255))
    return im

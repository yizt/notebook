# -*- coding: utf-8 -*-
"""
Created on 2019/3/30 下午2:50
几何图形相关

@author: mick.yi

"""
import cv2


def min_area_rect(polygon):
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    return rect, box

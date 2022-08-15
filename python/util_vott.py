# -*- coding: utf-8 -*-
"""
 @File    : util_vott.py
 @Time    : 2021/5/6 下午4:08
 @Author  : yizuotian
 @Description    :
"""
import json

import numpy as np


def parse_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        regions = list(ret_dic['assets'].values())[0]['regions']
    pts = np.zeros(shape=(len(regions), 4, 2))  # [N,(lt,lr,br,bl),(x,y)]
    for i, region in enumerate(regions):
        for j, pt in enumerate(region['points']):
            pts[i, j, 0] = pt['x']
            pts[i, j, 1] = pt['y']

    return pts

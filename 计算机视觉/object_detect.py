# -*- coding: utf-8 -*-
"""
   File Name：     object_detect
   Description :  目标检测相关
   Author :       mick.yi
   date：          2019/1/28
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from matplotlib import patches
import colorsys


def random_colors(num, bright=True):
    """
    随机生成颜色
    :param num:
    :param bright:
    :return:  [(r,g,b),...]  列表，长度num
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num, 1, brightness) for i in range(num)]  # 色调（H），饱和度（S），明度（V）
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def iou_np(boxes_a, boxes_b):
    """
    numpy 计算IoU
    :param boxes_a: (N,4)
    :param boxes_b: (M,4)
    :return:  IoU (N,M)
    """
    # 扩维
    boxes_a = np.expand_dims(boxes_a, axis=1)  # (N,1,4)
    boxes_b = np.expand_dims(boxes_b, axis=0)  # (1,M,4)

    # 分别计算高度和宽度的交集
    min_y2 = np.where(boxes_a[..., 2] < boxes_b[..., 2], boxes_a[..., 2], boxes_b[..., 2])  # (N,M)
    max_y1 = np.where(boxes_a[..., 0] > boxes_b[..., 0], boxes_a[..., 0], boxes_b[..., 0])
    overlap_h = np.where(min_y2 < max_y1, 0, min_y2 - max_y1 + 1)

    min_x2 = np.where(boxes_a[..., 3] < boxes_b[..., 3], boxes_a[..., 3], boxes_b[..., 3])
    max_x1 = np.where(boxes_a[..., 1] > boxes_b[..., 1], boxes_a[..., 1], boxes_b[..., 1])
    overlap_w = np.where(min_x2 < max_x1, 0, min_x2 - max_x1 + 1)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0] + 1) * (boxes_a[..., 3] - boxes_a[..., 1] + 1)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0] + 1) * (boxes_b[..., 3] - boxes_b[..., 1] + 1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def iou_tf(gt_boxes, anchors):
    """
    tf 计算iou
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :param anchors: [M,(y1,x1,y2,x2)]
    :return: IoU [N,M]
    """
    gt_boxes = tf.expand_dims(gt_boxes, axis=1)  # [N,1,4]
    anchors = tf.expand_dims(anchors, axis=0)  # [1,M,4]
    # 交集
    intersect_w = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 3], anchors[:, :, 3]) -
                             tf.maximum(gt_boxes[:, :, 1], anchors[:, :, 1]))
    intersect_h = tf.maximum(0.0,
                             tf.minimum(gt_boxes[:, :, 2], anchors[:, :, 2]) -
                             tf.maximum(gt_boxes[:, :, 0], anchors[:, :, 0]))
    intersect = intersect_h * intersect_w

    # 计算面积
    area_gt = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1]) * \
              (gt_boxes[:, :, 2] - gt_boxes[:, :, 0])
    area_anchor = (anchors[:, :, 3] - anchors[:, :, 1]) * \
                  (anchors[:, :, 2] - anchors[:, :, 0])

    # 计算并集
    union = area_gt + area_anchor - intersect
    # 交并比
    iou = tf.divide(intersect, union, name='regress_target_iou')
    return iou


def display_boxes(image, boxes, class_ids, class_names,
                  scores=None, title="",
                  figsize=(16, 16), ax=None,
                  show_bbox=True,
                  colors=None, captions=None):
    """
    # 边框可视化
    :param image: 图像的numpy数组
    :param boxes: 边框[N,(y1,x1,y2,x2)]
    :param class_ids:
    :param class_names:
    :param scores:
    :param title:
    :param figsize:
    :param ax:
    :param show_bbox:
    :param colors: 颜色列表  [(r,g,b)...]
    :param captions: 边框标题列表
    :return:
    """
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # 生成随机颜色
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()


def main():
    colors = random_colors(30)
    print(colors)


if __name__ == '__main__':
    main()

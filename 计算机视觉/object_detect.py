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
    overlap_h = np.maximum(0.0,
                           np.minimum(boxes_a[..., 2], boxes_b[..., 2]) -
                           np.maximum(boxes_a[..., 0], boxes_b[..., 0]))  # (N,M)

    overlap_w = np.maximum(0.0,
                           np.minimum(boxes_a[..., 3], boxes_b[..., 3]) -
                           np.maximum(boxes_a[..., 1], boxes_b[..., 1]))  # (N,M)

    # 交集
    overlap = overlap_w * overlap_h

    # 计算面积
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou


def iou_np_bak(boxes_a, boxes_b):
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


def gen_base_anchors(base_size, scales, ratios):
    """
    生成基准anchors
    :param base_size: 基准尺寸
    :param scales: 尺寸 (N,)
    :param ratios: 长宽比 (M,)
    :return:
    """
    scales = np.expand_dims(scales, axis=1)  # (N,1)
    ratios = np.expand_dims(ratios, axis=0)  # (1,M)

    # 分别计算长宽
    h = base_size * scales * np.sqrt(ratios)  # (N,M)
    w = base_size * scales / np.sqrt(ratios)

    # reshape为(N*M,)
    h = np.reshape(h, newshape=(-1,))
    w = np.reshape(w, newshape=(-1,))

    # 返回相对于原点的4个边框坐标
    return np.stack([-0.5 * h, -0.5 * w, 0.5 * h, 0.5 * w], axis=1)


def shift(base_anchors, shift_shape, strides):
    """
    生成所有anchors
    :param base_anchors: 基准anchors，[N,(y1,x1,y2,x2)]
    :param shift_shape: tuple或list (h,w), 最终的anchors个数为h*w*N
    :param strides:tuple或list (stride_h,stride_w),
    :return:
    """
    # 计算中心点在原图坐标
    center_y = strides[0] * (0.5 + np.arange(shift_shape[0]))
    center_x = strides[1] * (0.5 + np.arange(shift_shape[1]))

    center_x, center_y = np.meshgrid(center_x, center_y)  # (h,w)

    center_x = np.reshape(center_x, (-1, 1))  # (h*w,1)
    center_y = np.reshape(center_y, (-1, 1))  # (h*w,1)

    dual_center = np.concatenate([center_y, center_x, center_y, center_x], axis=1)  # (h*w,4)

    # 中心点和基准anchor合并
    base_anchors = np.expand_dims(base_anchors, axis=0)  # (1,N,4)
    dual_center = np.expand_dims(dual_center, axis=1)  # (h*w,1,4)
    anchors = base_anchors + dual_center  # (h*w,N,4)
    # 打平返回
    return np.reshape(anchors, (-1, 4))  # (h*w,4)


def nms(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=0.05, name=None):
    """
    非极大抑制
    :param boxes: 形状为[num_boxes, 4]的二维浮点型Tensor.
    :param scores: 形状为[num_boxes]的一维浮点型Tensor,表示与每个框(每行框)对应的单个分数.
    :param max_output_size: 一个标量整数Tensor,表示通过非最大抑制选择的框的最大数量.
    :param iou_threshold: 浮点数,IOU 阈值
    :param score_threshold:  浮点数, 过滤低于阈值的边框
    :param name:
    :return: 检测边框、边框得分
    """
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold, name)  # 一维索引
    output_boxes = tf.gather(boxes, indices)  # [M,4]
    class_scores = tf.gather(scores, indices)  # [M]
    return output_boxes, class_scores


def regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(y1,x1,y2,x2)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = tf.log(gt_h / h)
    dw = tf.log(gt_w / w)

    target = tf.stack([dy, dx, dh, dw], axis=1)
    target /= tf.constant([0.1, 0.1, 0.2, 0.2])
    # target = tf.where(tf.greater(target, 100.0), 100.0, target)
    return target


def np_regress_target(anchors, gt_boxes):
    """
    计算回归目标
    :param anchors: [N,(y1,x1,y2,x2)]
    :param gt_boxes: [N,(y1,x1,y2,x2)]
    :return: [N,(y1,x1,y2,x2)]
    """
    # 高度和宽度
    h = anchors[:, 2] - anchors[:, 0]
    w = anchors[:, 3] - anchors[:, 1]

    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    # 中心点
    center_y = (anchors[:, 2] + anchors[:, 0]) * 0.5
    center_x = (anchors[:, 3] + anchors[:, 1]) * 0.5
    gt_center_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) * 0.5
    gt_center_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) * 0.5

    # 回归目标
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = np.log(gt_h / h)
    dw = np.log(gt_w / w)

    target = np.stack([dy, dx, dh, dw], axis=1)
    target /= np.array([0.1, 0.1, 0.2, 0.2])
    return target


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
    # colors = random_colors(30)
    # print(colors)
    # anchors = gen_base_anchors(16, [1.0, 2.0, 3], [1.0, 1 / 2, 2])
    # print(anchors)
    boxa = np.array([[1, 1, 10, 10], [2, 2, 9, 9]], dtype=np.float32)
    boxb = np.array([[1, 1, 10, 9], [2, 2, 9, 9]], dtype=np.float32)
    # iou = iou_np(boxa, boxb)
    # print(iou)
    # iou = iou_np_bak(boxa, boxb)
    # print(iou)
    rgss = np_regress_target(boxb, boxa)
    print(rgss)


if __name__ == '__main__':
    main()

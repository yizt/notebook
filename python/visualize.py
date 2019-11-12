# -*- coding: utf-8 -*-
"""
   File Name：     visualize
   Description :  可视化
   Author :       mick.yi
   date：          2019/3/19
"""

import matplotlib.pyplot as plt
from matplotlib import patches, lines
import random
import numpy as np
import colorsys
import cv2


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_boxes(image, boxes, scores, class_names=None, colors=None):
    if len(boxes) == 0:
        return
    if colors is None:
        colors = random_colors(len(boxes))
    if class_names is None:
        class_names = [] * len(boxes)

    for box, score, class_name, color in zip(boxes.astype(np.int32), scores, class_names, colors):
        y1, x1, y2, x2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=4)

        text_str = '{}:{:.0%}'.format(class_name, score)
        cv2.putText(image, text_str, (x1, y1 - 8), cv2.FONT_HERSHEY_PLAIN, .75, color,
                    thickness=1)


def put_text(image, x, y, text, color=None):
    """
    写中文字
    :param image:
    :param x:
    :param y:
    :param text:
    :param color:
    :return:
    """
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(image)
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc')
    draw = ImageDraw.Draw(im)
    color = (255, 0, 0) if color is None else color
    draw.text((x, y), text, color, font=font)
    return np.array(im)


def put_texts(image, xs, ys, texts, color=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(image)
    font = ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc')
    draw = ImageDraw.Draw(im)
    color = (255, 0, 0) if color is None else color
    for x, y, text in zip(xs, ys, texts):
        draw.text((x, y), text, color, font=font)
    return np.array(im)


# 边框可视化
def display_boxes(image, boxes, class_ids, class_names,
                  scores=None, title="",
                  figsize=(16, 16), ax=None, show_bbox=True,
                  colors=None, captions=None):
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

    # Generate random colors
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


def display_polygon(image, polygons, scores=None, figsize=(16, 16), ax=None, colors=None):
    auto_show = False
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True
    if colors is None:
        colors = random_colors(len(polygons))

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    for i, polygon in enumerate(polygons):
        color = colors[i]
        polygon = np.reshape(polygon, (-1, 2))  # 转为[n,(x,y)]
        patch = patches.Polygon(polygon, facecolor=None, fill=False, color=color)
        ax.add_patch(patch)
        # 多边形得分
        x1, y1 = polygon[0][0], polygon[0][1]
        ax.text(x1, y1 + 8, scores[i] if scores is not None else '',
                color='w', size=11, backgroundcolor="none")
    if auto_show:
        plt.show()

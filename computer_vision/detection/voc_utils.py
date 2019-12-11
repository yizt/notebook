# -*- coding: utf-8 -*-
"""
 @File    : voc_utils.py
 @Time    : 2019/12/11 下午4:22
 @Author  : yizuotian
 @Description    :
"""
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
from lxml import etree, objectify

Annotation = namedtuple('Annotation',
                        'image_name image_path height width '
                        'gt_boxes class_names')


def save_to_xml_file(image_name, height, width, class_names, gt_boxes, dst_dir):
    E = objectify.ElementMaker(annotate=False)
    boxes = []
    for class_name, box in zip(class_names, gt_boxes):
        y1, x1, y2, x2 = box
        boxes.append(E.object(
            E.name(class_name),
            E.pose('Unspecified'),
            E.truncated('0'),
            E.difficult('0'),
            E.bndbox(
                E.xmin(x1),
                E.ymin(y1),
                E.xmax(x2),
                E.ymax(y2)
            )
        ))

    annotate_tree = E.annotation(
        E.folder('Annotation'),
        E.filename(image_name),
        E.source(
            E.database('Unknown'),
            E.annotation('PASCAL VOC2007'),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(3)
        ),
        E.segmented(0),
        *boxes
    )
    dst_xml_name = os.path.splitext(image_name)[0] + '.xml'
    dst_path = os.path.join(dst_dir, dst_xml_name)
    etree.ElementTree(annotate_tree).write(dst_path, pretty_print=True)


def parse_voc(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()

    objects = element.findall('object')
    image_name = element.find('filename').text
    width = int(element.find('size').find('width').text)
    height = int(element.find('size').find('height').text)

    gt_boxes = []
    class_names = []
    for element_obj in objects:
        class_name = element_obj.find('name').text
        obj_bbox = element_obj.find('bndbox')
        x1 = float(obj_bbox.find('xmin').text)
        y1 = float(obj_bbox.find('ymin').text)
        x2 = float(obj_bbox.find('xmax').text)
        y2 = float(obj_bbox.find('ymax').text)
        gt_boxes.append([y1, x1, y2, x2])
        class_names.append(class_name)

    return image_name, height, width, class_names, gt_boxes


def main():
    annotate = Annotation(image_name='abc.jpg',
                          image_path='abc.jpg',
                          height=800,
                          width=400,
                          class_names=['a', 'b'],
                          gt_boxes=np.random.randn(2, 4) * 200)
    save_to_xml_file('abc.jpg', 880, 400, ['a', 'b'], np.random.randn(2, 4) * 200, './')
    print(parse_voc('abc.xml'))


if __name__ == '__main__':
    main()

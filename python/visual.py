#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:08:16 2019

@author: yj
"""

import os
import cv2
import xml.dom.minidom
from tqdm import tqdm

image_path = "./JPEGImages/"
annotation_path = "./Annotations/"


def cv2_rectangle(img, xmin_data, ymin_data, xmax_data, ymax_data, label_name):
    label_list = ["fangxiangpan", "gangsisheng", "luntai", "yeyasanreqi", "zhitui", "zhuanxiangyouguan"]
    color_list = [(255, 0, 0), (255, 155, 55), (0, 255, 0), (55, 255, 155), (0, 0, 255), (155, 55, 255)]
    cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), color_list[label_list.index(label_name)], 5)
    # for i in range(len(lable_list)):
    #     if label_name == lable_list[i]:
    #         cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), color_list[l], 2)


files_name = os.listdir(image_path)
files_name.sort()
for filename_ in tqdm(files_name):
    filename, extension = os.path.splitext(filename_)
    # filename=filename.split('__')[0]
    img_path = image_path + filename_  # + '.jpg'
    xml_path = annotation_path + filename + '.xml'
    print("img_path", img_path)
    print("xml_path", xml_path)
    print("=" * 33)
    img = cv2.imread(img_path)
    # print("img_ori", img.shape)
    img_shape_ori = img.shape
    if img is None:
        pass
    try:
        dom = xml.dom.minidom.parse(xml_path)
    except:
        # os.remove(img_path)
        continue
    root = dom.documentElement
    pic_size = dom.getElementsByTagName("size")[0]
    pic_width = pic_size.getElementsByTagName('width')[0]
    pic_width = int(float(pic_width.childNodes[0].data))
    pic_height = pic_size.getElementsByTagName('height')[0]
    pic_height = int(float(pic_height.childNodes[0].data))
    pic_depth = pic_size.getElementsByTagName('depth')[0]
    pic_depth = int(float(pic_depth.childNodes[0].data))

    # row=height=y;colum=width=x
    objects = dom.getElementsByTagName("object")
    for object in objects:
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]

        xmin_data = int(float(xmin.childNodes[0].data))
        ymin_data = int(float(ymin.childNodes[0].data))
        xmax_data = int(float(xmax.childNodes[0].data))
        ymax_data = int(float(ymax.childNodes[0].data))
        print("xmin,ymin,xmax,ymax", xmin_data, ymin_data, xmax_data, ymax_data)
        # print("xmin<xmax", xmin_data < xmax_data)
        # print("ymin<ymax", ymin_data < ymax_data)
        label_name = object.getElementsByTagName('name')[0].childNodes[0].data
        print(label_name)

        if label_name == 'fangxiangpan':
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (255, 0, 0), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)
        elif label_name == 'gangsisheng':
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (255, 155, 55), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 155, 55), 5)
        elif label_name == 'luntai':
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (0, 255, 0), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 5)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
        elif label_name == 'yeyasanreqi':
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (55, 255, 155), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (55, 255, 155), 5)
        elif label_name == 'zhitui':
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (0, 0, 255), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        else:
            cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (155, 55, 255), 5)
            cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (155, 55, 255), 5)

    #    cv2_rectangle(img, xmin_data, ymin_data, xmax_data, ymax_data, label_name)
    #         if label_name =='hand':
    #             cv2.rectangle(img,(xmin_data,ymin_data),(xmax_data,ymax_data),(55,255,155),2)
    # #            cv2.putText(img,label_name,(int((xmin_data+xmax_data/2)),int((ymin_data+ymax_data)/2)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    #         elif label_name == 'phone':
    #             cv2.rectangle(img,(xmin_data,ymin_data),(xmax_data,ymax_data),(255,155,55),2)
    #         else:
    #             cv2.rectangle(img,(xmin_data,ymin_data),(xmax_data,ymax_data),(155,55,255),2)
    flag = 0
    flag = cv2.imwrite("./Visualization-test/{}.jpg".format(filename), img)
    xml_shape = (pic_height, pic_width, pic_depth)
    if (pic_height, pic_width, pic_depth) != img_shape_ori:
        print((pic_height, pic_width, pic_depth), img_shape_ori)
    # print("img_ori and img_rectangle", img.shape, img_shape_ori == img.shape)
    # (h,w) = image.shape[:2]
    # print("xml_shape",(pic_height,pic_width,pic_depth),(pic_height,pic_width,pic_depth)==img_shape_ori)
    if not (flag):
        print(filename, "error")
print("all done ====================================")

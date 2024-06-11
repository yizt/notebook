# -*- coding: utf-8 -*-
"""
 @File    : test_segment_voc.py
 @Time    : 2022/7/15 上午11:22
 @Author  : yizuotian
 @Description    :
"""
import os

import cv2
import numpy as np


def show_voc_data(data_dir="/Users/yizuotian/dataset/mixed_align"):
    """
    查看voc数据
    :param data_dir:
    :return:
    """
    im = 'D02_20210829221834_455'
    npy_path = os.path.join(data_dir, 'data_dataset_voc/SegmentationClass/{}.npy'.format(im))
    png_path = os.path.join(data_dir, 'data_dataset_voc/SegmentationClassPNG/{}.png'.format(im))
    gray_png_path = os.path.join(data_dir, 'SegmentationClassRaw/{}.png'.format(im))

    npy = np.load(npy_path)
    png = cv2.imread(png_path)
    gray_png = cv2.imread(gray_png_path)

    print(npy.shape)
    print(png.shape)
    print(gray_png.shape)

    print(np.max(npy), np.max(png), np.max(gray_png))
    print(np.unique(png))


def split_dataset(image_dir, list_dir):
    from glob import glob
    from random import shuffle
    # 以上是利用灰度图（标注数据）
    images = glob(image_dir+"/*")
    shuffle(images)

    total_num = len(images)
    print("总数量：", total_num)

    train_num = int(total_num * 0.9)
    print("训练集数量：", train_num)
    print("验证集数量：", total_num - train_num)

    train_path = os.path.join(list_dir,'train.txt')
    val_path = os.path.join(list_dir,'val.txt')
    list_file_1 = open(train_path, 'w')
    for item1 in images[:train_num]:
        image_id = item1.split('/')[-1].replace(".jpg", "").replace(".png", "")
        list_file_1.write('%s\n' % image_id)
    list_file_1.close()

    list_file_2 = open(val_path, 'w')
    for item2 in images[train_num:]:
        image_id = item2.split('/')[-1].replace(".jpg", "").replace(".png", "")
        list_file_2.write('%s\n' % image_id)
    list_file_2.close()


if __name__ == '__main__':
    # show_voc_data()
    split_dataset("/Users/yizuotian/dataset/mixed_align/data_dataset_voc/JPEGImages",
                  "/Users/yizuotian/dataset/mixed_align/data_dataset_voc/ImageSets/Segmentation")

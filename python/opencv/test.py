# -*- coding: utf-8 -*-
"""
 @File    : test.py
 @Time    : 2020/3/21 上午8:51
 @Author  : yizuotian
 @Description    :
"""

import math

import cv2
import numpy as np
from scipy import misc, ndimage


def fun1():
    filepath = './'

    img = cv2.imread('abc.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    rotate_angle = 0
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if x1 == x2 or y1 == y2:
            continue
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
    print("rotate_angle : " + str(rotate_angle))
    rotate_img = ndimage.rotate(img, rotate_angle)
    misc.imsave('ssss.png', rotate_img)
    cv2.imshow("img", rotate_img)
    cv2.waitKey(0)


def func2(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        return
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)
    misc.imsave('abc_out.jpg', rotate_img)

# func2('abc.jpeg')


import numpy as np
import os
import cv2
import math
from scipy import misc, ndimage


def rotate(image,angle,center=None,scale=1.0):
    (w,h) = image.shape[0:2]
    if center is None:
        center = (w//2,h//2)
    wrapMat = cv2.getRotationMatrix2D(center,angle,scale)
    return cv2.warpAffine(image,wrapMat,(h,w))
#使用矩形框
def getCorrect(img_path):
    #读取图片，灰度化
    src = cv2.imread(img_path)
    # cv2.imshow("src",src)
    cv2.waitKey()
    gray = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    cv2.imshow("gray",gray)
    cv2.waitKey()
    #图像取非
    grayNot = cv2.bitwise_not(gray)
    cv2.imshow("grayNot",grayNot)
    cv2.waitKey()
    #二值化
    threImg = cv2.threshold(grayNot,100,255,cv2.THRESH_BINARY,)[1]
    cv2.imshow("threImg",threImg)
    cv2.waitKey()
    #获得有文本区域的点集,求点集的最小外接矩形框，并返回旋转角度
    coords = np.column_stack(np.where(threImg>0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(angle + 90)
    else:
        angle = -angle

    #仿射变换，将原图校正
    dst = rotate(src,angle)
    cv2.imshow("dst",dst)
    cv2.waitKey()
    print(angle)


def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


# 使用霍夫变换
def getCorrect2(img_path):
    # 读取图片，灰度化
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    showAndWaitKey("src", src)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    showAndWaitKey("gray", gray)
    # 腐蚀、膨胀
    kernel = np.ones((5, 5), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)
    showAndWaitKey("eroDil", eroDil)
    # 边缘检测
    canny = cv2.Canny(eroDil, 50, 150)
    showAndWaitKey("canny", canny)
    # 霍夫变换得到线条
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength=100, maxLineGap=10)
    drawing = np.zeros(src.shape[:], dtype=np.uint8)
    # 画出线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    showAndWaitKey("houghP", drawing)
    """
    计算角度,因为x轴向右，y轴向下，所有计算的斜率是常规下斜率的相反数，我们就用这个斜率（旋转角度）进行旋转
    """
    k = float(y1 - y2) / (x1 - x2)
    thera = np.degrees(math.atan(k))

    """
    旋转角度大于0，则逆时针旋转，否则顺时针旋转
    """
    rotateImg = rotate(src, thera)
    cv2.imshow("rotateImg", rotateImg)
    cv2.waitKey()


def showAndWaitKey(winName, img):
    cv2.imshow(winName, img)
    cv2.waitKey()


if __name__ == "__main__":
    getCorrect('abc.jpeg')

# -*- coding: utf-8 -*-
"""
 @File    : test_np.py
 @Time    : 2020/8/26 下午4:53
 @Author  : yizuotian
 @Description    :
""" 
import numpy as np

x = np.ones([2,3,3])

x = np.random.rand(2,3)
np.sum(x)
np.sum(x,axis=(0,1))
np.sum(x,axis=0)

print(np.max(x,axis=(0,1)))
print(x)
print(np.argmax(x,axis=0))
print(np.argmax(x))
print(np.argmax(x.flatten()))
np.argsort(x)
np.sort

np.flatten

np.ravel

np.squeeze

np.split

np.dot

np.matmul



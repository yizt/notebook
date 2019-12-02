# -*- coding: utf-8 -*-
"""
 @File    : ncnn.py
 @Time    : 2019/12/2 下午5:11
 @Author  : yizuotian
 @Description    :
"""


from torchvision import models
import torch
import torch.onnx



def main():
    x = torch.rand(1, 3, 300, 300)
    m = models.resnet50(pretrained=False)
    m.load_state_dict(torch.load('/Users/yizuotian/pretrained_model/resnet50-19c8e357.pth'))
    torch_out = torch.onnx._export(m, x, "resnet50.onnx", export_params=True)


if __name__ == '__main__':
    main()
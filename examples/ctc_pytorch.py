# -*- coding: utf-8 -*-
"""
 @File    : ctc_pytorch.py
 @Time    : 2020/10/22 上午8:00
 @Author  : yizuotian
 @Description    :
"""
import itertools
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class TextImage(object):
    def __init__(self, alpha, im_h, im_w, max_chars, min_chars):
        """

        :param alpha:
        :param im_h:
        :param im_w:
        :param max_chars:
        :param min_chars:
        """
        self.alpha = alpha
        self.im_h = im_h
        self.im_w = im_w
        self.max_chars = max_chars
        self.min_chars = min_chars

    def gen(self):
        char_len = np.random.randint(self.min_chars, self.max_chars + 1)
        chars = np.random.choice(list(self.alpha), char_len)
        text = ''.join(chars)
        im = np.ones((self.im_h, self.im_w), dtype=np.uint8) * 255
        cv2.putText(im, text=text,
                    org=(np.random.randint(self.im_w // len(text)),
                         int(self.im_h * 0.7 + np.random.rand() * 3)),
                    fontFace=np.random.choice([1, 5, 6, 7]),
                    fontScale=1. + 0.1 * np.random.randn(),
                    color=np.random.randint(150), thickness=1)
        return im, text


def norm_im(im_gray):
    img = np.transpose(im_gray[:, :, np.newaxis], axes=(2, 1, 0))  # [H,W] => [C,W,H]
    # 标准化
    image = img.astype(np.float32) / 255.
    image -= np.mean(image)
    return image


class Generator(Dataset):
    def __init__(self, text_image: TextImage, **kwargs):
        self.text_image = text_image
        self.alpha = text_image.alpha
        self.max_len = self.text_image.max_chars
        self.input_len = int(self.text_image.im_w / 4 - 3)
        super(Generator, self).__init__(**kwargs)

    def __getitem__(self, index):
        """

        :param index:
        :return img: [C,W,H] 灰度图
        :return target: [max_len] GT字符类别id
        :return input_len: 输入的长度
        :return target_len: GT字符个数
        """
        im_gray, text = self.text_image.gen()
        image = norm_im(im_gray)

        indices = [self.text_image.alpha.index(char) for char in text]
        indices = np.array(indices) + 1  # 0 留给默认空白
        target = np.zeros(shape=(self.max_len,), dtype=np.long)
        target_len = len(text)
        target[:target_len] = indices

        return image, target, self.input_len, target_len

    def __len__(self):
        return 2000


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=False):
        super(_ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if bn:
            self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))


class CrnnSmall(nn.Module):
    """
    CRNN简化版
    """

    def __init__(self, num_classes, num_base_filters=16, **kwargs):
        super(CrnnSmall, self).__init__(**kwargs)
        self.cnn = nn.Sequential(OrderedDict([
            ('conv_block_1', _ConvBlock(1, num_base_filters)),  # [B,c,W,32]
            ('max_pool_1', nn.MaxPool2d(2, 2)),  # [B,c,W/2,16]

            ('conv_block_2', _ConvBlock(num_base_filters, num_base_filters * 2)),  # [B,2c,W/2,16]
            ('max_pool_2', nn.MaxPool2d(2, 2)),  # [B,128,W/4,8]

            ('conv_block_3_1', _ConvBlock(num_base_filters * 2, num_base_filters * 4)),  # [B,4c,W/4,8]
            ('max_pool_3', nn.MaxPool2d((2, 2), (1, 2))),  # [B,256,W/8,4]

            ('conv_block_4_1', _ConvBlock(num_base_filters * 4, num_base_filters * 8, bn=True)),  # [B,8c,W/8,4]
            ('max_pool_4', nn.MaxPool2d((2, 2), (1, 2))),  # [B,512,W/16,2]

            ('conv_block_5', _ConvBlock(num_base_filters * 8, num_base_filters * 8,
                                        kernel_size=(2, 2), padding=0))  # [B,8c,W/16,1]
        ]))

        self.rnn1 = nn.GRU(num_base_filters * 8, num_base_filters * 4, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(num_base_filters * 8, num_base_filters * 4, batch_first=True, bidirectional=True)
        self.transcript = nn.Linear(num_base_filters * 8, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        x = self.cnn(x)  # [B,8c,W/16,1]
        x = torch.squeeze(x, 3)  # [B,8c,W]
        x = x.permute([0, 2, 1])  # [B,W,8c]
        x, h1 = self.rnn1(x)
        x, h2 = self.rnn2(x, h1)
        x = self.transcript(x)
        return x


def test_im_gen():
    text_image = TextImage('abc', 32, 96, 5, 2)
    for _ in range(10):
        im, _ = text_image.gen()
        cv2.imshow('a', im)
        cv2.waitKeyEx(0)


def main():
    text_image = TextImage('abc', 32, 96, 5, 2)
    data_set = Generator(text_image)
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    model = CrnnSmall(len(text_image.alpha) + 1, num_base_filters=8)
    criterion = CTCLoss()

    optimizer = optim.Adadelta(model.parameters(), weight_decay=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for image, target, input_len, target_len in tqdm(data_loader):
            # print(target, target_len, input_len)
            outputs = model(image.to(torch.float32))  # [B,N,C]
            m_outputs = outputs
            outputs = torch.log_softmax(outputs, dim=2).to(torch.float64)
            outputs = outputs.permute([1, 0, 2])  # [N,B,C]
            loss = criterion(outputs[:], target, input_len, target_len)
            # 梯度更新
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * image.size(0)
            if np.isnan(loss.item()):
                print(target, m_outputs)

        epoch_loss = epoch_loss / len(data_loader.dataset)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, num_epochs, epoch_loss))

    # 预测结果


def inference(net, alpha, image_gray):
    net.eval()
    image = norm_im(image_gray)
    image = torch.FloatTensor(image)

    predict = torch.softmax(net(image)[0], dim=-1)  # 求置信度需要
    predict = predict.detach().cpu().numpy()  # [W,num_classes]
    label = np.argmax(predict[:], axis=1) - 1
    label = [alpha[class_id] if class_id >= 0 else '' for class_id in label]
    label = [k for k, g in itertools.groupby(list(label))]
    label = ''.join(label)
    return label


if __name__ == '__main__':
    main()
    # test_im_gen()

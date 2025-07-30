#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-27 14:58:22
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-27 14:58:23
@ FilePath: /mqtt_examples/gmm_torch/gmm_wx.py
@ Description: Do edit!
"""

from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from gmm_torch.ggmm_custom import QuaternionMultivariateNormal,QuaternionNormal,QuaternionGMM


def train():
    # 生成示例数据（假设有3个高斯分布）
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=[0]*7, scale=0.5, size=(100, 7)),
        np.random.normal(loc=[3]*7, scale=0.5, size=(100, 7)),
        np.random.normal(loc=[-3]*7, scale=0.5, size=(100, 7))
    ])
    data = torch.tensor(data, dtype=torch.float32)

    # 创建DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 超参数设置
    n_components = 3
    input_dim = 7
    learning_rate = 0.01
    n_epochs = 100

    # 初始化模型和优化器
    model = QuaternionGMM(n_components)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    3
    # 训练循环
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            # logger.debug(batch)
            # 前向传播
            log_prob = model(x)
            loss = -log_prob.max(dim=1).values.mean()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / len(data)
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')

    # 获取最终参数
    params = model.get_parameters()
    print("Learned parameters:")
    print(f"pi: {params['pi']}")
    print(f"mu: {params['mu']}")
#    print(f"var: {params['var']}")

    logger.debug(f"{model.score_samples(data[1:3])}")

def vis(data,n_components,params):
    # 创建网格用于可视化
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    # 绘制数据点和等高线
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5)

    for k in range(n_components):
        mu = params['mu'][k]
        var = params['var'][k]
        
        # 绘制椭圆（假设协方差矩阵为对角矩阵）
        from matplotlib.patches import Ellipse
        width, height = 2 * np.sqrt(5.991 * var)  # 95%置信区间
        angle = 0
        ellipse = Ellipse(mu, width, height, angle, 
                        facecolor='none', edgecolor='r', linewidth=2)
        plt.gca().add_patch(ellipse)

    plt.title('GMM Clustering')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__=='__main__':
    train()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-29 10:14:45
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-29 10:14:46
@ FilePath: /mqtt_examples/gmm_torch/gmm_dk.py
@ Description: Do edit!
"""
from loguru import logger
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal

def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    # 定义可训练参数
    loc = torch.randn(2, requires_grad=True)
    # 使用 Cholesky 分解参数化协方差矩阵
    scale_tril = torch.tril(torch.randn(2, 2))
    scale_tril.diagonal(dim1=-2, dim2=-1).abs_()  # 确保对角线为正
    scale_tril.requires_grad_(True)

    # 定义优化器
    optimizer = optim.Adam([loc, scale_tril], lr=0.01)

    for _ in range(100):
        logger.debug(_)
        optimizer.zero_grad()
        scale_tril.data = scale_tril.tril()
        # 构造多元正态分布
        mvn = MultivariateNormal(loc, scale_tril=scale_tril)
        
        # 计算损失（示例：最大化样本的对数似然）
        samples = mvn.rsample((100,))  # 生成样本
        loss = -mvn.log_prob(samples).mean()  # 负对数似然
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 可选：强制 scale_tril 保持正定性
        with torch.no_grad():
            scale_tril.diagonal(dim1=-2, dim2=-1).abs_()

def torch_test():
    x = torch.randn(3, 3)
    logger.debug(f"{x=}")

    y = torch.tril(x)
    logger.debug(f"{y=}")
    z = y.diagonal(dim1=-1, dim2=-2)
    logger.debug(f"{z=}")
    logger.debug(f"{y.diagonal(dim1=-2, dim2=-1)=}")
    u = z.abs_()
    logger.debug(f"{u=}")

    logger.debug(f'{x.max(dim=-1)=}')
    logger.debug(f'{x.max(dim=-1).values=}')
    logger.debug(f"{x.diag()=}")
    logger.debug(f"{x.diagonal().copy_(x.diag().abs())=}")
    logger.debug(f"{x=}")

    matrix = torch.randn(3, 3)
    # 确保矩阵是正定的
    from torch.distributions.constraints import positive_definite
    constrained_matrix = positive_definite.check(matrix)
    logger.debug(f"{constrained_matrix=}")
    logger.debug(f"{matrix.size(0)=}")
    tril = torch.tril(torch.ones(6,6))
    logger.debug(f"{tril=}")



if __name__ == "__main__":
	# test()
    torch_test()

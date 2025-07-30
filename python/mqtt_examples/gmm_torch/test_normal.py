#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-27 15:08:41
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-27 15:08:41
@ FilePath: /mqtt_examples/gmm_torch/test_normal.py
@ Description: Do edit!
"""
import torch
from torch.distributions import Normal,MultivariateNormal
from loguru import logger
def test():
	"""
	@ description: 
	@ param {type} 
	@ return: 
	"""
	m = Normal(torch.tensor([0.0,1.]), torch.tensor([1.0,1.]))
	logger.debug(f"{torch.exp(m.log_prob(torch.Tensor([0,1])))=}")
	
	mean = torch.tensor([0.0, 0.0])  # 均值向量 (2维)
	cov_matrix = torch.tensor([[1.0, 0], 
							[0, 1.0]])  # 协方差矩阵 (必须是对称正定矩阵)
	
	# 创建多元正态分布对象
	mvn = MultivariateNormal(mean, cov_matrix)
	logger.debug(f"{torch.exp(mvn.log_prob(torch.Tensor([0,1])))=}")
    # m = MultivariateNormal()
	logger.debug(torch.randn((3,4)).max(dim=-1).values.mean())
	logger.debug(torch.randn((3,4)).max(dim=-1).mean())
	logger.debug(torch.arange(1,300,3).argmax())

if __name__ == "__main__":
	test()

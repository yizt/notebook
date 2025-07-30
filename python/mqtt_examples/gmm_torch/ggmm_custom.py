#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-27 16:19:50
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-27 16:19:51
@ FilePath: /mqtt_examples/gmm_torch/custom_gmm.py
@ Description: Do edit!
"""
from torch import nn
from loguru import logger
from torch.distributions import MultivariateNormal,Normal
from torch.distributions.multivariate_normal import _batch_mahalanobis
import math
import torch
import torch.nn.functional as F
import math
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all


def quaternion_multiply(q1, q2):  
    # q1, q2: 四元数，形状为 (..., 4)，其中最后一维为 [w, x, y, z]  
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]  
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]  
      
    # 四元数乘法公式  
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  
      
    return torch.stack([w, x, y, z], dim=-1)  

def quat_mult(q1, q2):
    """
    四元素乘法
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_log(q):
    """Logarithm map from quaternion to axis-angle (batch version)."""
    q = F.normalize(q,p=2,dim=-1)
    sign = torch.sign(q[..., 0:1])
    sign[sign == 0] = 1.0
    q = q * sign
    w = q[..., 0]
    vec = q[..., 1:]
    norm_vec = torch.norm(vec, dim=-1, keepdim=True)
    small = norm_vec.squeeze(-1) < 1e-10
    theta = 2 * torch.acos(torch.clamp(w, -1 + 1e-8, 1 - 1e-8))
    axis = vec / (norm_vec + 1e-15)
    log = theta.unsqueeze(-1) * axis
    log[small] = 0.0
    return log

def quat_conjugate(q):
    """
    四元素的共轭值
    @param q: [N,(w, rx, ry, rz)] or [(w, rx, ry, rz)]
    """
    conjugate = q.clone()
    conjugate[...,1:] = q[...,1:]*-1
    return conjugate

def distance(x,u):
    """
    计算样本与均值的距离
    @param x:[N,(x,y,z, w, rx, ry, rz)]
    @param u:均值 [(x,y,z, w, rx, ry, rz)]
    """
    pos_diff = x[:, :3] - u[:3]
    quat_product = quat_mult(x[...,3:], quat_conjugate(u)[...,3:])
    quat_diff = quat_log(quat_product)
    return torch.cat([pos_diff, quat_diff], dim=-1)

def quat_weighted_mean(quaternions, weights, eps=1e-8):
    """Weighted quaternion mean (batch version)."""
    weights = torch.nan_to_num(weights, nan=eps)
    weights = torch.clamp(weights, min=0) + eps
    weights /= weights.sum()

    sign = torch.sign(quaternions[:, 0:1])
    sign[sign == 0] = 1.0
    signed_q = quaternions * sign

    w_sqrt = torch.sqrt(weights).unsqueeze(-1)
    q_weighted = signed_q * w_sqrt
    C = q_weighted.t() @ q_weighted
    C += torch.eye(4, device=C.device) * eps

    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    mean_q = eigenvectors[:, torch.argmax(eigenvalues)]
    return F.normalize(mean_q,dim=-1)


class QuaternionNormal(Normal):
    def __init__(self, mean, loc, scale, validate_args=None):
        self.p_mean = mean
        super().__init__(loc,scale,validate_args)

    def log_prob(self, value):
        # if self._validate_args:
        #     self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return (
            # -((value - self.loc) ** 2) / (2 * var)
            -(distance(value, self.p_mean) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

class QuaternionMultivariateNormal(MultivariateNormal):
    """
    四元素的多元高斯分布
    """
    def __init__(
        self,
        mean,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        self.p_mean = mean
        super().__init__(loc,covariance_matrix,precision_matrix,scale_tril,validate_args)
        
    def log_prob(self, value):
        # if self._validate_args:
        #     self._validate_sample(value)
        # diff = value - self.loc
        diff = distance(value, self.p_mean)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det


class QuaternionGMM(nn.Module):
    def __init__(self, n_components):
        super(QuaternionGMM, self).__init__()
        self.n_components = n_components
        input_dim = 7   # pos+quat
        
        # 初始化混合系数（使用softmax保证和为1）
        self.pi = nn.Parameter(torch.ones(n_components) / n_components)
        
        # 初始化均值
        self.mu = nn.Parameter(torch.Tensor(n_components, input_dim))
        self.mu.data.normal_()  # 从标准正态分布初始化
        
        # 初始化协方差矩阵（使用对角矩阵简化计算）
        self.loc = torch.zeros(size=(n_components,input_dim-1))
        
        # self.log_var = nn.Parameter(torch.eye(input_dim-1).repeat(n_components, 1, 1))
        # self.log_var.data.fill_(0)  # 初始方差为exp(-1) ≈ 0.3679
        self.scale_tril = nn.Parameter(torch.tril(torch.randn(6, 6)).abs().repeat(n_components, 1, 1))
        self.register_buffer('tril_sign',torch.tril(torch.ones(6, 6)))
        

    def forward(self, x):
        # 计算每个分布的概率密度
        n_samples = x.size(0)
        log_prob = torch.zeros(n_samples, self.n_components)
        
        for k in range(self.n_components):
            # 计算多元高斯分布的概率密度
            # var = torch.exp(self.log_var[k])
            #logger.debug(f"{var=}")
            # dist = QuaternionNormal(self.mu[k], self.loc[k], var.sqrt())
            scale_tril = self.scale_tril[k].clone().detach()
            logger.debug(f"before {scale_tril=}")
            scale_tril = scale_tril.tril()
            scale_tril.diagonal().copy_(scale_tril.diag().abs())
            logger.debug(f"before {scale_tril=}")
            self.scale_tril[k].data = scale_tril # 确保上三角
            
            dist = QuaternionMultivariateNormal(self.mu[k], self.loc[k], scale_tril=self.scale_tril[k])
            #logger.debug(f"{dist.log_prob(x)=}")
            log_prob[:, k] = dist.log_prob(x) + torch.log(self.pi[k])
        
        # 计算对数似然
        # log_likelihood = torch.logsumexp(log_prob, dim=1)
        # # return -log_likelihood.mean()  # 返回平均负对数似然作为损失
        # logger.debug(f"{log_likelihood.shape=}")
        # logger.debug(f"{log_prob.shape=}")
        return log_prob

    def get_parameters(self):
        return {
            'pi': self.pi.detach().numpy(),
            'mu': self.mu.detach().numpy(),
            # 'var': torch.exp(self.log_var).detach().numpy()
            'scale_tril': self.scale_tril.detach().numpy()
        }

    def predict(self, x):
        log_prob = self.forward(x)
        return torch.argmax(log_prob, dim=-1)

    def predict_proba(self, x):
        """
        softmax概率归一化
        """
        log_prob = self.forward(x)
        return torch.exp(log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True))

    def score_samples(self, x):
        return self.forward(x).max(dim=-1)



def test():
    q1 = torch.tensor([[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 3.0, 4.0]])  # 四元数 1: w=1, x=2, y=3, z=4  
    q2 = torch.tensor([[5.0, 6.0, 7.0, 8.0],[1.0, 2.0, 3.0, 4.0]])  # 四元数 2: w=5, x=6, y=7, z=8  
    
    logger.debug(f"四元数乘法结果:{quaternion_multiply(q1, q1)=}" )
    # logger.debug(f"四元数乘法结果:{quat_mult(q1, q2)=}")
    logger.debug(f"{quat_conjugate(q1[0])=}")
    logger.debug(f"{F.normalize(q1,p=2,dim=-1)=}")
	

if __name__ == "__main__":
	test()

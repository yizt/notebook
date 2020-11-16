# -*- coding: utf-8 -*-
"""
 @File    : transformer.py
 @Time    : 2020/11/16 下午3:05
 @Author  : yizuotian
 @Description    :
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def attention(query, key, value, mask=None, dropout=None):
    """
    普通的注意力
    :param query: [B,N,dk] [batch_size,sequence length,dimension length]
    :param key: [B,N,dk]
    :param value: [B,N,dv]
    :param mask: [B,N,N]
    :param dropout :
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B,N,N]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # [B,N,N]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "生成n个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    """
    多头注意力,a)降维分组，b)每个分组使用普通的dot-scale注意力;c)结果concat,d)再做一个fc
    """

    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: 分组数
        :param d_model: q,k,v的维度长度
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # [B,N,N] => [B,1,N,N]
        nbatches = query.size(0)

        # [B,N,d_model] => [B,h,N,dk]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # dot-scale attention
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # [B,h,N,dk] => [B,N,h,dk] => [B,N,d_model]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # 最后线性变换
        return self.linears[-1](x)


class SelfAttention(nn.Module):
    """
    多头自注意力从头开始实现
    """

    def __init__(self, hid_dim, n_heads, dropout):
        """

        :param hid_dim: q,k,v原始维度
        :param n_heads: head数
        :param dropout:
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        """

        :param query: [B,N,hid_dim]
        :param key: [B,N,hid_dim]
        :param value: [B,N,hid_dim]
        :param mask: [B,N,N]
        :return:
        """
        # Q,K,V计算与变形：

        bsz = query.shape[0]
        # [B,N,hid_dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # [B,N,hid_dim]=> [B,N,n_heads,d] =>[B,n_heads,N,d]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [B,n_heads,N,N]

        # 如果没有mask，就生成一个
        if mask is not None:
            mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：

        attention = self.do(torch.softmax(energy, dim=-1))  # [B,n_heads,N,N]

        # 第三步，attention结果与V相乘

        x = torch.matmul(attention, V)  # [B,n_heads,N,d]

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()  # [B,N,n_heads,d]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))  # [B,N,hid_dim]

        x = self.fc(x)  # [B,N,hid_dim]

        return x

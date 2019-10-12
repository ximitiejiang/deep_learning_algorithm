#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:22:44 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
"""损失函数库：
1. 采用独立函数定义损失函数的底层实现，但不包括权重和缩减
2. 采用nn.Module类的形式，统一forward的接口(pred, target, weight, avg_factor)
   同时处理损失的权重和缩减问题

"""

# %%
class CrossEntropyLoss(nn.Module):
    """多类别交叉熵：带softmax和独热编码变换模块在内部
    args:
        pred: 任意
        target： 任意
    """
    def __init__(self):
        pass
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        # 初始化权重
        if weight is None:
            weight = torch.ones_like(pred)
        # 初始化平均因子
        if avg_factor is None:
            avg_factor = torch.tensor([1])
        # 计算损失
        loss = weight * F.cross_entropy(pred, target, reduction='none') / avg_factor
        return loss


class BinaryCrossEntropyLossWithLogits(nn.Module):
    """二值交叉熵：带sigmoid函数在内部
    args
        pred: (b, n_class)任意
        target: (b, n_class)必须是[0,1]且为2列的独热编码形式
    """    
    def __init__(self):
        pass
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        # 初始化权重
        if weight is None:
            weight = torch.ones_like(pred)
        # 初始化平均因子
        if avg_factor is None:
            avg_factor = torch.tensor([1])
        # 计算损失
        loss = weight * F.binary_cross_entropy_with_logits(
                pred, target, reduction='none') / avg_factor
        return loss


# %% 
class SigmoidFocalLoss(nn.Module):
    """多类别focalloss：内部自带sigmoid
    注意：类别数目上不能包含背景类别，即voc(20), coco(80)
    args:
        pred: (b, n_class)任意数据
        target: (b, n_class)必须是n_class列的独热编码形式
    """
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        if weight is None:
            weght = torch.ones_like(pred)
        if avg_factor is None:
            avg_factor = 1
        loss = weight * focal_loss(pred, target, self.alpha, self.gamma)
        loss = loss.sum() / avg_factor
        return loss

def focal_loss(pred, target, alpha, gamma):
    pred_sigmoid = pred.sigmoid()
    pt
    at
       

# %%
class SmoothL1Loss(nn.Module):
    """回归损失：柔性l1"""
    def __init__(self, beta, reduction):
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target, weight, avg_factor):
        loss = weight * smooth_l1_loss(pred, target, self.beta, reduction='none')
        loss = loss.sum() / avg_factor
        return loss

def smooth_l1_loss(pred, target, beta):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

# %%
class IouLoss(nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, pred, target, weight, avg_factor):
        pass


def iou_loss(pred, target):
    pass        
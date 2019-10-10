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
    """分类损失：交叉熵"""
    def __init__(self):
        pass
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = weight * F.cross_entropy()


# %%
def focal_loss(pred, target, alpha, gamma):
    pred_sigmoid = pred.sigmoid()
    pt
    at
        
class FocalLoss(nn.Module):
    
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

# %%
def smooth_l1_loss(pred, target, beta):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


class SmoothL1Loss(nn.Module):
    """回归损失：柔性l1"""
    def __init__(self, beta, reduction):
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target, weight, avg_factor):
        loss = weight * smooth_l1_loss(pred, target, self.beta)
        loss = loss.sum() / avg_factor
        return loss


# %%
def iou_loss(pred, target):
    pass

class IouLoss(nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, pred, target, weight, avg_factor):
        pass
        
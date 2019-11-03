#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:22:44 2019

@author: ubuntu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ious import calc_ious_tensor

"""损失函数库：
1. 采用独立函数定义损失函数的底层实现，但不包括权重和缩减
2. 采用nn.Module类的形式，统一forward的接口(pred, target, weight, avg_factor)
   同时处理损失的权重和缩减问题

"""

def softmax(x):
    """numpy版本softmax函数, x(m,)为一维数组"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 在np.exp(x - C), 相当于对x归一化，防止因x过大导致exp(x)无穷大使softmax输出无穷大 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# %%
class CrossEntropyLoss(nn.Module):
    """多类别交叉熵：带softmax和独热编码变换模块在内部
    args:
        pred: 任意
        target： 任意
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = F.cross_entropy(pred, target, reduction='none') 
        if weight is not None:
            loss = weight * loss   
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    """二值交叉熵：带sigmoid函数在内部
    args
        pred: (b, n_class)任意
        target: (b, n_class)
    """    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        if target.dim() != pred.dim():
            raise ValueError('target should be onehot code, with same dim as predicts.')
        loss = F.binary_cross_entropy_with_logits(
                    pred, target, reduction='none')   
        if weight is not None:
            loss = weight * loss          
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss


# %% 
class SigmoidFocalLoss(nn.Module):
    """多类别focalloss：内部自带sigmoid，所以输入pred不需要预先做sigmoid处理。
    Focal loss的基本原理：在原有交叉熵基础上
    注意：类别数目上不能包含背景类别，即voc(20), coco(80)
    args:
        pred: (b, n_class)任意数据
        target: (b, n_class)必须是n_class列的独热编码形式
    """
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = focal_loss(pred, target, self.alpha, self.gamma) 
        
        if weight is not None:
            loss = weight * loss      
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """focal loss底层函数: alpha是正负样本比例控制参数，gamma是难样本控制参数
    focal_loss = - at*(1-pt)^gamma * log(pt), 其中at = a*t+(1-a)(1-t), pt = p*t+(1-p)(1-t)
    当t=1时
    """
    pred_s = pred.sigmoid()  # 用真实概率p进行weight的计算，但二值交叉熵输入非概率化p，因为内部自带了sigmoid
    pt = pred_s * target + (1 - pred_s) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    weight = at * (1 - pt).pow(gamma)
    loss = F.binary_cross_entropy(pred, target, weight, reduction='none')
    return loss


# %%
class SmoothL1Loss(nn.Module):
    """回归损失：柔性l1"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weight=None, avg_factor=None):        
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        if weight is not None:
            loss = weight * loss      
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss


def smooth_l1_loss(pred, target, beta=1.):
    """柔性l1 loss底层函数, 用作参考，但底层实际还是采用pytorch的F函数库
    args:
        pred: (k,4)
        target: (k,4)
    returns:
        loss: (k,)
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


# %%
class IouLoss(nn.Module):
    """iou loss基础上增加centerness作为权重"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, weight, avg_factor):
        loss = iou_loss(pred, target)
        if weight is not None:
            loss = weight * loss      
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss

def iou_loss(pred, target):
    """iou loss底层函数，loss = -log(iou)作为回归损失函数
    args:
        pred: (k,4)
        target: (k,4)
    return: 
        loss: (k,)
    """
    ious = calc_ious_tensor(pred, target, aliged=True).clamp(min=1e-6)
    loss = - ious.log()
    return loss        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:35:34 2019

@author: ubuntu
"""
import torch
import torch.nn.functional as F
from utils.ious import calc_ious_tensor

def smooth_l1_loss(pred, target, beta=1.0, reduction='elementwise_mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 0:
        return loss
    elif reduction == 1:
        return loss.sum() / pred.numel()
    elif reduction == 2:
        return loss.sum()


def weighted_smooth_l1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight) / avg_factor  # 这里修改了下让输出是一个tensor标量

# %% focal loss
def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='elementwise_mean'):
    """带sigmoid的focal loss实现，focal loss主要采用二值交叉熵方式对多分类问题进行损失计算。
    focal loss的核心是采用二值交叉熵来处理多分类任务，所以必须采用0,1形式独热编码输入。
    同时在类别数量上只需要考虑正类别(如20类，80类)而不需要考虑背景类，因为背景类在每一个二分类中作为0是考虑的。
    同时相关优化都体现在weight权重的创建上：focal_loss = bce_loss(p, t, at * pt)
        - 其中at = a * t + (1 - a)*(1 - t), 当t=0负样本时，at=1-a, 当t=1正样本时，at=a, a一般取0.25
        - 其中pt = p * t + (1 - p)*(1 - t), 当t=0负样本时，pt=1-p, 当t=1正样本时，pt=p
        - 所以整个weight = at * (1 - pt)**gamma, 
        
    注意：
    1. 输入的pred可以是任意形式，内部会增加sigmoid
    2. 输入的多分类label需要事先转为独热编码标签(这是binary_cross_entropy_with_logits所要求的)
    args:
        pred: (m, 20)
        target: (m, 20)
        weight: (m, 20)
        reduction:
    """
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)  # pt = (1-p)*
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    return F.binary_cross_entropy_with_logits(
        pred, target, weight, reduction=reduction)


    
    
def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=20):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


# %%
    
#def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
#    """Apply element-wise weight and reduce loss.
#
#    Args:
#        loss (Tensor): Element-wise loss.
#        weight (Tensor): Element-wise weights.
#        reduction (str): Same as built-in losses of PyTorch.
#        avg_factor (float): Avarage factor when computing the mean of losses.
#
#    Returns:
#        Tensor: Processed loss values.
#    """
#    # if weight is specified, apply element-wise weight
#    if weight is not None:
#        loss = loss * weight
#
#    # if avg_factor is not specified, just reduce the loss
#    if avg_factor is None:
#        loss = reduce_loss(loss, reduction)
#    else:
#        # if reduction is mean, then average the loss by avg_factor
#        if reduction == 'mean':
#            loss = loss.sum() / avg_factor
#        # if reduction is 'none', then do nothing, otherwise raise an error
#        elif reduction != 'none':
#            raise ValueError('avg_factor can not be used with reduction="sum"')
#    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None,
                          num_classes=20):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    # 先计算loss(不缩减)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    # 然后缩减loss        
#    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    if avg_factor is None:  # 如果平均因子为none，则用
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    loss = loss.sum() / avg_factor
    return loss


# %%
def iou_loss(pred, target):
    """用来计算一组预测bboxes跟真实bbox之间坐标的回归损失
    之前采用smooth_l1, 即用距离等于两个坐标的差作为度量，
    这里采用loss = -log(iou)作为度量
    args:
        pred: (m, 4)
        target: (m, 4)
    
    """
    ious = calc_ious_tensor(pred, target, aliged=True)     # 注意：这里是采用一对一求iou, (m,)
    loss = - ious.log()
    return loss
    
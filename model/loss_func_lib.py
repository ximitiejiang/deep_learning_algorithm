#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:35:34 2019

@author: ubuntu
"""
import torch
import torch.nn.functional as F


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
    """带sigmoid的focal loss实现：
    args:
        pred:
        target:
        weight:
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
    
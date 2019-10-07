#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:23:55 2019

@author: ubuntu
"""
import torch

def miou(preds, labels):
    """用来计算分割任务的精度指标
    args:
        preds: tensor(b, n_class, h, w)
        labels: tensor(b, h, w)
    """
    pass
    

def pixel_acc(preds, labels):
    """用来计算分割任务的精度指标：
    args:
        preds: tensor(b, n_class, h, w)
        labels: tensor(b, h, w)        
    """
    # preds预测值范围(0,20)调整为(1,21)这样方便计算总数
    preds = torch.argmax(preds.long(), dim=1) + 1
    labels = labels.long() + 1
    
    pixel_labels = torch.sum(labels > 0)
    pixel_correct
    return pixel_correct, pixel_labeled


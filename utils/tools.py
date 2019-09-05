#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:43:44 2019

@author: ubuntu
"""
import torch

def accuracy(y_pred, label, topk=1):
    """pytorch tensor版本的精度计算：由于都是未概率化的数据，
    y_pred(b, n_classes)，tensor为未概率化的数据
    label(b,), tensor为未概率化的数据(也就是实际标签而不是独热编码)
    输出: acc (float标量)
    """
    with torch.no_grad():
        # TODO: 增加topk的功能
        if topk == 1:
            pred = torch.argmax(y_pred, dim=1) # 输出(b,)
            acc = (pred == label).sum().float() / len(label)
        return acc
    

import time
def get_time_str():
    """计算系统时间并生成字符串"""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())
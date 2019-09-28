#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:55:55 2019

@author: ubuntu
"""

# %% 
import torch
import numpy as np

def calc_ious_tensor(bboxes1, bboxes2):
    """cuda tensor版本的iou计算
    args:
        bboxes1(m,4)
        bboxes2(n,4)
    output:
        ious(m,n)
    """
    bb1 = bboxes1
    bb2 = bboxes2
    xymin = torch.max(bb1[:, None, :2], bb2[:, :2])
    xymax = torch.min(bb1[:, None, 2:], bb2[:, 2:])
    
    wh = (xymax - xymin + 1).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]  # (m,n)
    area1 = (bb1[:, 2] - bb1[:, 0] + 1) * (bb1[:, 3] - bb1[:, 1] + 1) # (m,)
    area2 = (bb2[:, 2] - bb2[:, 0] + 1) * (bb2[:, 3] - bb2[:, 1] + 1) # (n,)
    
    ious = area / (area1[:, None] + area2 - area)  # (m,n)/((m,1)+(n,)-(m,n)) = (m,n) 
    return ious

        
def calc_ious_np(bboxes1, bboxes2):
    """numpy版计算两组bboxes中每2个bbox之间的iou(包括所有组合，而不只是位置对应的bbox)
    bb1(m, 4), bb2(n, 4), 假定bb1是gt_bbox，则每个gt_bbox需要跟所有anchor计算iou，
    也就是提取每一个gt，因此先从bb1也就是bb1插入轴，(m,1,4),(n,4)->(m,n,4)，也可以先从bb2插入空轴则得到(n,m,4)"""
    # 在numpy环境操作(也可以用pytorch)
    bb1 = bboxes1.numpy()
    bb2 = bboxes2.numpy()
    # 计算重叠区域的左上角，右下角坐标
    xymin = np.max(bb1[:, None, :2] , bb2[:, :2])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    xymax = np.min(bb1[:, 2:] , bb2[:, None, 2:])  # (m,2)(n,2) -> (m,1, 2)(n,2) -> (m,n,2)
    # 计算重叠区域w,h
    wh = xymax - xymin # (m,n,2)-(m,n,2) = (m,n,2)
    # 计算重叠面积和两组bbox面积
    area = wh[:, :, 0] * wh[:, :, 1] # (m,n)
    area1 = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]) # (m,)*(m,)->(m,)
    area2 = (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]) # (n,)*(n,)->(n,)
    # 计算iou
    ious = area / (area1[:, None] + area2 - area)     #(m,n) /[(m,)+(1,n)-(m,n)] -> (m,n) / (m,n)
    
    return ious  # (m,n)
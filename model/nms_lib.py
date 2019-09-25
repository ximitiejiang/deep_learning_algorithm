#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 12:12:19 2019

@author: ubuntu
"""

import torch
import numpy as np

"""
nms = dict(
        type = 'nms',
        score_thr = 0.02,    # 最小score： 目的是筛除大部分空bbox
        max_per_img = 200,   # 最大bbox个数
        params = dict(
                iou_thr=0.5) # 重叠度大于0.5的bbox则去除
        )

"""

def nms_operation(bboxes, scores, type, score_thr=0.02, max_per_img=200, params=None):
    """对输出bbox坐标和bbox得分进行过滤，同时生成预测标签：
    bbox坐标的过滤主要通过iou，bbox得分的过滤主要通过设置得分阈值
    注意：这里采用的是对每一个类分别筛选，便于后续的分类别处理。
    args:
        bboxes:(n_anchors, 4)
        scores:(n_anchors, 21)
        type: 表示nms类型, nms或soft_nms
        score_thr: score的阈值
        max_per_img: 一张图上最多输出多少bbox
        params: 表示nms对象需要的参数，主要提供iou_thr=0.5，表示大于?
    """
    num_classes = scores.shape[1]
    # 获得nms操作对象：可以是nms， soft_nms
    nms_op = get_nms_op(type)
    
    nmsed_bboxes = []
    nmsed_labels = []
    # 分别评估一张图片结果的每一个类
    for i in range(1, num_classes): # 1-20
        # 筛除大部分空bbox
        cls_inds = scores[:, i] > score_thr
        _bboxes = bboxes[cls_inds, :]    #(k, 4)
        _scores = scores[cls_inds, :]    #(k, )
        # 组合bbox和score
        bbox_and_score = torch.cat([_bboxes, _scores.reshape(-1, 1)], dim=1) # (n, 5)
        # 执行nms: 过滤重叠bbox
        bbox_and_score, _ = nms_op(bbox_and_score, **params)  # (k, 5)
        cls_labels = bboxes.new_full((bbox_and_score.shape[0], ), i, dtype=torch.long)  # (k,)赋值对应标签值为(1-20)中的一类
        # 保存nms结果
        nmsed_bboxes.append(bbox_and_score)
        nmsed_labels.append(cls_labels)
    # 如果有输出结果
    if nmsed_bboxes:
        nmsed_bboxes = torch.cat(nmsed_bboxes)
        nmsed_labels = torch.cat(nmsed_labels)
        # 如果输出结果超过上线，则取得分最高的前一部分
        if nmsed_bboxes.shape[0] > max_per_img:
            _, inds = nmsed_bboxes[:, -1].sort(descending=True)
            inds = inds[:max_per_img]
            nmsed_bboxes = nmsed_bboxes[inds]
            nmsed_labels = nmsed_labels[inds]
    # 如果没有结果，则输出空
    else:
        nmsed_bboxes = bboxes.new_zeros((0, 5))
        nmsed_labels = bboxes.new_zeros((0,), dtype=torch.long)
    
    return nmsed_bboxes, nmsed_labels  # (k, 5)包含bbox坐标和置信度，(k,)包含标签
    

def get_nms_op(nms_type):
    nms_dict = {
            'nms': nms}
    nms_func = nms_dict[nms_type]
    return nms_func

# %%
"""如何创建cython代码加速nms计算
1. 创建gpu_nms.pyx

2. 创建gpu_nms.hpp


3. 创建Makefile

"""

#from model.nms.gpu_nms import gpu_nms
#from model.nms.cpu_nms import cpu_nms
#from model.cpu_soft_nms import cpu_soft_nms
from v01.utils.nms.gpu_nms import gpu_nms
from v01.utils.nms.cpu_nms import cpu_nms

def nms(preds, iou_thr):
    """基本版nms: 链接到cpu_nms或gpu_nms, 
    如果是numpy则用cpu_nms，而如果是tensor，则链接到gpu_nms
    args:
        preds: (k, 5)包括4个坐标和1个置信度score
        iou_thr: 重叠阈值，高于该阈值则代表两个bbox重叠，删除其中score比较低的
    """
    # 进行格式变换
    if isinstance(preds, torch.Tensor):
        device_id = preds.get_device()  # 获得输入的设备号， cpu=-1, cuda= 1~n
        preds_np = preds.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        device_id = None
        preds_np = preds   
    # 进行nms: 需要采用numpy送入函数
    if device_id is not None:  # 如果是tensor
        inds = gpu_nms(preds_np, iou_thr, device_id=device_id)
        inds = preds.new_tensor(inds, dtype=torch.long)  # 恢复tensor
    elif device_id is None:    # 如果是numpy
        inds = cpu_nms(preds_np, iou_thr)
    return preds[inds, :], inds
    

#def soft_nms(preds, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
#    """soft nms: 比nms更容易
#    args:
#        preds: (k, 5)包括4个坐标和1个置信度score
#        iou_thr: 重叠阈值，高于该阈值则代表两个bbox重叠，删除其中score比较低的
#        method:
#        sigma:
#        min_score:
#    """
#    if isinstance(preds, torch.Tensor):
#        device_id = preds.get_device()  # 获得输入的设备号， cpu=-1, cuda= 1~n
#        preds_np = preds.detach().cpu().numpy()
#    elif isinstance(preds, np.ndarray):
#        device_id = None
#        preds_np = preds
#    # 进行soft nms: 需要送入numpy
#    new_preds, inds = cpu_soft_nms(preds_np, iou_thr, method=method_codes[method],
#                                   sigma=sigma, min_score=min_score)
#    if device_id is not None: # 如果是tensor
#        new_preds = preds.new_tensor(new_preds)
#        inds = preds.new_tensor(inds)
#        return new_preds, inds
#    else:
#        return new_preds.astype(np.float32), inds.astype(np.int64)
        













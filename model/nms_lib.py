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
        type = 'nms',        # soft_nms, nms, simple
        score_thr = 0.02,    # 最小score： 目的是筛除大部分空bbox
        max_per_img = 200,   # 最大bbox个数
        params = dict(
                iou_thr=0.5) # 重叠度大于0.5的bbox则去除
        )
"""
def nms_wrapper2(bboxes, scores, ldmks=None, type=None, score_thr=None, max_per_img=None, params=None):
    """重写nms wrapper: nms目的是基于bbox和score进行重复bbox的筛除
    args:
        bboxes(m, 4)
        scores(m, n_cls): 这里包含了背景0类和从1-n的前景类
        type: nms类型，可选nms, softnms, nms_debug，softnms_debug
        score_thr: score阈值
        max_per_img: 一张图最多输出多少bbox
        params: nms操作参数，包括iou_thr=0.5
    """
    bbox_outs, ldmk_outs, label_outs = [], [], []
    if ldmks is None:
        ldmks = bboxes.new_zeros((0, 0, 2))
    n_cls = scores.shape[1]
    for i in range(1, n_cls):  # 按类别
        # score过滤
        cls_inds = scores[:, i] > score_thr
        bboxes = bboxes[cls_inds, :]  # (n, 4)
        scores = scores[cls_inds, i]  # (n,)
        ldmks = ldmks[cls_inds, :]    # (n,5,2)
        # nms过滤
        dets = torch.cat([bboxes, scores.reshape(-1,1)], dim=1) #(n,5)
        keep = nms_op(dets, **params, type=type)
        dets = dets[keep, :]
        ldmks = ldmks[keep, :]
        labels = dets.new_full((dets.shape[0], ), i, dtype=torch.long)  # (n,) 从1到n_cls-1
        # 保存
        bbox_outs.append(dets)
        ldmk_outs.append(ldmks)
        label_outs.append(labels)
#    if bbox_outs:  # 是不是应该用len(bbox_outs) > 0
#        # 合并所有类
#        bbox_outs = torch.cat(bbox_outs, dim=0).reshape(-1, 5)
#        ldmk_outs = torch.cat(ldmk_outs, dim=0)
#        label_outs = torch.cat(label_outs, dim=0)
#        if bbox_outs.shape[0] > max_per_img:
#            _, inds = bbox_outs[:, -1].sort(descending=True)
#            inds = inds[:max_per_img]
#            bbox_outs = bbox_outs[inds]
#            label_outs = label_outs[inds]
#            ldmk_outs = ldmk_outs[inds]
#    else:
#        bbox_outs = bbox_outs.new_zeros((0, 5))
#        label_outs = label_outs.new_zeros((0,), dtype=torch.long)
#        ldmk_outs = ldmk_outs.new_zeros((0,10))
    return bbox_outs, label_outs, ldmk_outs   # (n_cls,)(m,5),  (n_cls,)(m,),  (n_cls,)(m,5,2) 
        

def nms_op(dets, iou_thr, type=None):
    """用来定义nms
    args:
        dets(m,5)
        type: nms, softnms, nms_debug，softnms_debug 其中debug版本表示用纯python/cpu版本的nms做调试用
    returns:
        keep(n,) 代表n个保留的bbox的index
    """
    if type is None:
        type = 'nms'
    # 格式变换：都是用numpy格式(cython的要求)
    if isinstance(dets, torch.Tensor):
        device_id = dets.get_device()  # 获得输入的设备号， cpu=-1, cuda= 0~n
        dets_np = dets.detach().cpu().numpy()    
    else:
        raise ValueError('input data should be tensor type.')
    
    # 选择nms
    if dets_np.shape[0] == 0:
        keep = []

    if type == 'nms_debug':
        keep = py_cpu_nms(dets, iou_thr)
    elif type == 'softnms_debug':
        keep = py_cpu_soft_nms(dets, iou_thr)    

    elif type == 'softnms':
        from model.nms.cpu_soft_nms import cpu_soft_nms
        keep = cpu_soft_nms(dets, iou_thr)

    elif type == 'nms':
        if device_id >= 0:  # 如果是gpu tensor
            from model.nms.gpu_nms import gpu_nms
            keep = gpu_nms(dets_np, iou_thr, device_id=device_id)
        elif device_id == -1:    # 如果是cpu tensor
            from model.nms.cpu_nms import cpu_nms
            keep = cpu_nms(dets_np, iou_thr)
    keep = dets.new_tensor(keep, dtype=torch.long)  # 恢复tensor
    return keep
    
    
def py_cpu_nms(dets, iou_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep

def py_cpu_soft_nms(dets, iou_thr):
    pass


# %%
def nms_wrapper(bboxes, scores, type, score_thr=0.02, max_per_img=200, params=None):
    """ 这是nms wrapper, 对输出bbox坐标和bbox得分进行过滤，同时生成预测标签，且调用底层的nms函数。   
    1. nms wrapper操作过程：
        - bbox置信度的初筛(主要通过设置最低阈值)： 用于去除大部分空的bbox
        - bbox坐标的过滤(主要通过nms): 用于去除大部分重叠的bbox
        - bbox标签的获取：基于分类输出的列号来获得label
    2. nms的基本原理：
        - 
    3. 理解nms函数调用过程
        (1). 顶层：用python的nms wrapper获取数据，并根据设置来调用不同类型的nms函数
        (2). 中间层：用cython定义不同类型的nms函数，包括cpu_nms, gpu_nms, cpu_soft_nms
            - cython作为中间的粘合语言，既可以被python调用，又可以调用c/c++/cuda语言，
              且cython不调用别的语言，pyx生成的代码亦可以比python执行速度快数倍。
            - cython编写cpu版本的cpu_nms, cpu_soft_nms，由于变量优化，依然比python源码要快数倍
        (3). 底层：用cuda编写_nms函数给gpu_nms来调用
            - 编写.hpp头文件和.cu源文件，然后通过setup.py进行整体编译
            - setup.py文件的执行过程：
    
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
        _scores = scores[cls_inds, i]    #(k, )只提取该类的score
        # 组合bbox和score
        bbox_and_score = torch.cat([_bboxes, _scores.reshape(-1, 1)], dim=1) # (n, 5)
        # 执行nms: 过滤重叠bbox
        keep = nms_op(bbox_and_score, **params)  # (k, 5)
        bbox_and_score = bbox_and_score[keep, :]
        cls_labels = bboxes.new_full((bbox_and_score.shape[0], ), i, dtype=torch.long)  # (k,)赋值对应标签值为(1-20)中的一类
        # 保存nms结果
        nmsed_bboxes.append(bbox_and_score)
        nmsed_labels.append(cls_labels)
    # 如果有输出结果
    if nmsed_bboxes:
        nmsed_bboxes = torch.cat(nmsed_bboxes).reshape(-1, 5)
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
"""

"""
#from model.cpu_soft_nms import cpu_soft_nms

def nms(preds, iou_thr):
    """基本版nms: 链接到cpu_nms或gpu_nms, 
    如果是numpy则用cpu_nms，而如果是tensor，则链接到gpu_nms
    args:
        preds: (k, 5)包括4个坐标和1个置信度score
        iou_thr: 重叠阈值，高于该阈值则代表两个bbox重叠，删除其中score比较低的
    """
    # 进行格式变换
    if isinstance(preds, torch.Tensor):
        device_id = preds.get_device()  # 获得输入的设备号， cpu=-1, cuda= 0~n
        preds_np = preds.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        device_id = None
        preds_np = preds   
    # 进行nms: 需要采用numpy送入函数
    if preds_np.shape[0] == 0:
        keep = []
    elif device_id is not None and device_id >= 0:  # 如果是gpu tensor
        from model.nms.gpu_nms import gpu_nms
        keep = gpu_nms(preds_np, iou_thr, device_id=device_id)
        keep = preds.new_tensor(keep, dtype=torch.long)  # 恢复tensor
    elif device_id is None or device_id == -1:    # 如果是numpy或cpu tensor
        from model.nms.cpu_nms import cpu_nms
        keep = cpu_nms(preds_np, iou_thr)
    return keep
    













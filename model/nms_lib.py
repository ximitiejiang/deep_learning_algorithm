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
        type = 'nms',        # softnms, nms, nms_dubug, softnms_debug
        score_thr = 0.02,    # 最小score： 目的是筛除大部分空bbox
        max_per_img = 200,   # 最大bbox个数
        params = dict(
                iou_thr=0.5) # 重叠度大于0.5的bbox则去除
        )
"""
def nms_wrapper(bboxes, scores, ldmks=None, type=None, score_thr=None, max_per_img=None, params=None):
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
    for i in range(1, n_cls):  # 按类别: 但不再考虑背景的筛选
        # score过滤
        cls_inds = scores[:, i] > score_thr
        _bboxes = bboxes[cls_inds, :]  # (n, 4)
        _scores = scores[cls_inds, i]  # (n,)
        _ldmks = ldmks[cls_inds, :] if len(ldmks) > 0 else ldmks    # (n,5,2)
        # nms过滤
        _dets = torch.cat([_bboxes, _scores.reshape(-1,1)], dim=1) #(n,5)
        keep = nms_op(_dets, **params, type=type)
        _dets = _dets[keep, :]
        _ldmks = _ldmks[keep, :] if len(_ldmks) > 0 else _ldmks
        _labels = _dets.new_full((_dets.shape[0], ), i, dtype=torch.long)  # (n,) 从1到n_cls-1
        # 保存
        bbox_outs.append(_dets)
        ldmk_outs.append(_ldmks)
        label_outs.append(_labels)
    return bbox_outs, label_outs, ldmk_outs   # (n_cls,)(m,5),  (n_cls,)(m,),  (n_cls,)(m,5,2) 
        

def nms_op(dets, iou_thr, type=None):
    """用来定义选择具体nms op.
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
    elif type == 'nms_debug':
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
    

# %% 底层nms操作  
    
def py_cpu_nms(dets, iou_thr):
    """纯python的cpu nms"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 核心就是一句话，排序提取最大值，然后剩余的筛除超过iou阈值的部分，然后循环
                                    # 所以核心就是获取order，搬到keep, 然后直到order搬空
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

def py_cpu_soft_nms(dets, iou_thr, sigma=0.5, Nt=0.5, method=2, threshold=0.1):
    """纯python的cpu soft nms"""
    box_len = len(dets)   # box的个数
    for i in range(box_len):
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        max_pos = i
        max_scores = ts

        # get max box
        pos = i+1
        while pos < box_len:
            if max_scores < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos
            pos += 1

        # add max box as a detection
        dets[i, :] = dets[max_pos, :]

        # swap ith box with position of max box
        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # 将置信度最高的 box 赋给临时变量
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]

        pos = i+1
        # NMS iterations, note that box_len changes if detection boxes fall below threshold
        while pos < box_len:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]

            area = (x2 - x1 + 1)*(y2 - y1 + 1)

            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih > 0:
                overlaps = iw * ih
                ious = overlaps / ((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1) + area - overlaps)

                if method == 1:    # 线性
                    if ious > Nt:
                        weight = 1 - ious
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ious**2) / sigma)
                else:              # original NMS
                    if ious > Nt:
                        weight = 0
                    else:
                        weight = 1

                # 赋予该box新的置信度
                dets[pos, 4] = weight * dets[pos, 4]

                # 如果box得分低于阈值thresh，则通过与最后一个框交换来丢弃该框
                if dets[pos, 4] < threshold:
                    dets[pos, 0] = dets[box_len-1, 0]
                    dets[pos, 1] = dets[box_len-1, 1]
                    dets[pos, 2] = dets[box_len-1, 2]
                    dets[pos, 3] = dets[box_len-1, 3]
                    dets[pos, 4] = dets[box_len-1, 4]

                    box_len = box_len-1
                    pos = pos-1
            pos += 1

    keep = [i for i in range(box_len)]
    return keep











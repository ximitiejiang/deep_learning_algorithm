import torch
from nms import nms_wrapper
import numpy as np

def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline. RGB大神在Faster rcnn上的代码
    逻辑：
    Args:
        dets:(m,5)  
        thresh:scaler
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  #[::-1]表示降序排序，输出为其对应序号

    keep = []                     #需要保留的bounding box
    while order.size > 0:
        i = order[0]              #取置信度最大的（即第一个）框
        keep.append(i)            #将其作为保留的框
        
        #以下计算置信度最大的框（order[0]）与其它所有的框（order[1:]，即第二到最后一个）框的IOU，以下都是以向量形式表示和计算
        xx1 = np.maximum(x1[i], x1[order[1:]]) #计算xmin的max,即overlap的xmin
        yy1 = np.maximum(y1[i], y1[order[1:]]) #计算ymin的max,即overlap的ymin
        xx2 = np.minimum(x2[i], x2[order[1:]]) #计算xmax的min,即overlap的xmax
        yy2 = np.minimum(y2[i], y2[order[1:]]) #计算ymax的min,即overlap的ymax

        w = np.maximum(0.0, xx2 - xx1 + 1)      #计算overlap的width
        h = np.maximum(0.0, yy2 - yy1 + 1)      #计算overlap的hight
        inter = w * h                           #计算overlap的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter) #计算并，-inter是因为交集部分加了两次。

        inds = np.where(ovr <= thresh)[0]          #本轮，order仅保留IOU不大于阈值的下标
        order = order[inds + 1]                    #删除IOU大于阈值的框, inds从1开始

    return keep  

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # bbox(m,5)最后一列score (xmin,ymi,xmax,ymax, score)
    
    def plot_bbox(dets, c='k'):
        x1 = dets[:,0]
        y1 = dets[:,1]
        x2 = dets[:,2]
        y2 = dets[:,3]
        plt.plot([x1,x2], [y1,y1], c)
        plt.plot([x1,x1], [y1,y2], c)
        plt.plot([x1,x2], [y2,y2], c)
        plt.plot([x2,x2], [y1,y2], c)
        plt.title("after nms")

    plot_bbox(boxes,'k')   # before nms
    keep = py_cpu_nms(boxes, thresh=0.7)
    plot_bbox(boxes[keep], 'r')# after nms
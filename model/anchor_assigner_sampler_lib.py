#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:50:42 2019

@author: ubuntu
"""
import numpy as np

# %% anchor身份指定器
    
class MaxIouAssigner():
    """用于指定每个anchor的身份是正样本还是负样本：基于anchor跟gt_bboxes的iou计算结果进行指定"""
    def __init__(self, pos_iou_thr, neg_iou_thr, min_pos_iou=0):
        self.pos_iou_thr = pos_iou_thr     #正样本阈值：大于该值则为正样本
        self.neg_iou_thr = neg_iou_thr     #负样本阈值：小于该值则为负样本
        self.min_pos_iou = min_pos_iou     #最小正样本阈值： 
        
    def assign(self, anchors, gt_bboxes, gt_labels):
        # 计算ious
        ious = calc_ious(gt_bboxes, anchors)  # (m,4)(n,4)->(m,n)
        anchor_maxiou_for_all_gt = ious.max(axis=0)          # (n,)
        anchor_maxiou_idx_for_all_gt = ious.argmax(axis=0)   # (n,) 0~n_gt
        
        #gt_maxiou_for_all_anchor = ious.max(axis=1)          # (m,)
        gt_maxiou_idx_for_all_anchor = ious.argmax(axis=1)   # (m,)
        
        # 基于规则指定每个anchor的身份, 创建anchor标识变量，先指定所有anchor为-1
        # 然后设置负样本=0，正样本=idx+1>0
        num_anchors = ious.shape[1]
        assigned_gt_inds = np.full((num_anchors, ), -1)      # (n, )
        # 小于负样本阈值，则设为0
        neg_idx = (anchor_maxiou_for_all_gt < self.neg_iou_thr) & (anchor_maxiou_for_all_gt >=0)
        assigned_gt_inds[neg_idx] = 0
        # 大于正样本阈值，则设为对应gt的index + 1 (>0)也代表gt的编号
        pos_idx = (anchor_maxiou_for_all_gt > self.pos_iou_thr) & (anchor_maxiou_for_all_gt <1)
        assigned_gt_inds[pos_idx] = anchor_maxiou_idx_for_all_gt[pos_idx] + 1 # 从0~k-1变到1～k,该值就代表了第几个gt   
        # 每个gt所对应的最大iou的anchor也设置为index + 1(>0)也代表gt的编号
        # 这样确保每个gt至少有一个anchor对应
        for i, anchor_idx in enumerate(gt_maxiou_idx_for_all_anchor):
            assigned_gt_inds[anchor_idx] = i + 1   # 从0~k-1变到1～k,该值就代表了第几个gt 
        
        # 转换正样本的标识从1~indx+1为真实gt_label
        assigned_gt_labels = np.zeros((num_anchors, ))     # (n, )
        for i, assign in enumerate(assigned_gt_inds):
            if assign > 0:
                label = gt_labels[assign-1]
                assigned_gt_labels[i] = label
        
        return [assigned_gt_inds, assigned_gt_labels, ious] # [(n,), (n,), (m,n)] 
            


# %% anchor采样器
        
class PseudoSampler():
    def __init__(self):
        pass
    
    def sample(self, assign_result, anchor_list, gt_bboxes):
        # 提取正负样本的位置号
        pos_inds = np.where(assign_result[0] > 0)[0]  #
        neg_inds = np.where(assign_result[0] == 0)[0]
        
#        pos_bboxes = 
        
        return [pos_inds, neg_inds]
        


# %% 
        
def calc_ious(bboxes1, bboxes2):
    """用于计算两组bboxes中每2个bbox之间的iou(包括所有组合，而不只是位置对应的bbox)
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
    ious = area / (area1 + area2[:,None,:] - area)     #(m,n) /[(m,)+(1,n)-(m,n)] -> (m,n) / (m,n)
    
    return ious  # (m,n)


 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:50:42 2019

@author: ubuntu
"""
import torch
from utils.ious import calc_ious_tensor



# %% anchor身份指定器
    
class MaxIouAssigner():
    """用于指定每个anchor的身份是正样本还是负样本：基于anchor跟gt_bboxes的iou计算结果进行指定
    outputs:
        assigned_gt_inds
        assigned_gt_labels
        ious
    """
    def __init__(self, pos_iou_thr, neg_iou_thr, min_pos_iou=0):
        self.pos_iou_thr = pos_iou_thr     #正样本阈值：大于该值则为正样本
        self.neg_iou_thr = neg_iou_thr     #负样本阈值：小于该值则为负样本
        self.min_pos_iou = min_pos_iou     #最小正样本阈值： 
        
    def assign(self, anchors, gt_bboxes, gt_labels):
        # 计算ious
        ious = calc_ious_tensor(gt_bboxes, anchors)  # (m,4)(n,4)->(m,n),其中gt的m在行，n在列
        # 获取
        anchor_maxiou_for_all_gt, anchor_maxiou_idx_for_all_gt = ious.max(dim=0)  # (n,) (n,) torch的max既返回值，又返回idx
        gt_maxiou_for_all_anchor, gt_maxiou_idx_for_all_anchor = ious.max(dim=1) # (m,) (m,)

        # 基于规则指定每个anchor的身份, 创建anchor标识变量，先指定所有anchor为-1 (然后设置负样本=0，正样本=idx+1>0)
        num_anchors = ious.shape[1]
        assigned_gt_inds = ious.new_full((num_anchors, ), -1, dtype=torch.long)  # (n, ) 借用ious的device(cuda)
        # 小于负样本阈值，则为负样本，设为0
        neg_idx = (anchor_maxiou_for_all_gt < self.neg_iou_thr) & (anchor_maxiou_for_all_gt >=0)
        assigned_gt_inds[neg_idx] = 0
        # 大于正样本阈值，则为正样本，则设为对应gt的index + 1 (>0)也代表gt的编号
        pos_idx = (anchor_maxiou_for_all_gt > self.pos_iou_thr) & (anchor_maxiou_for_all_gt <1)
        assigned_gt_inds[pos_idx] = anchor_maxiou_idx_for_all_gt[pos_idx] + 1 # 从0~k-1变到1～k,该值就代表了第几个gt   
        # 每个gt所对应的最大iou的anchor也设置为index + 1(>0)也代表gt的编号
        # 这样确保每个gt至少有一个anchor对应
        for i, anchor_idx in enumerate(gt_maxiou_idx_for_all_anchor):
            assigned_gt_inds[anchor_idx] = i + 1   # 从0~k-1变到1～k,该值就代表了第几个gt 
        
        # 转换正样本的标识从1~indx+1为真实gt_label
        if gt_labels is not None:
            assigned_gt_labels = ious.new_full((num_anchors, ), 0, dtype=torch.long) # (n, )
            for i, assign in enumerate(assigned_gt_inds):
                if assign > 0:
                    label = gt_labels[assign-1]
                    assigned_gt_labels[i] = label
                else:
                    continue
        else:
            assigned_gt_labels = None
        
        return [assigned_gt_inds, assigned_gt_labels, ious] # [(n,), (n,), (m,n)] 
            


# %% anchor采样器
        
class PseudoSampler():
    def __init__(self):
        pass
    
    def sample(self, assigned_gt_inds, assigned_gt_labels, ious, anchor_list, gt_bboxes):
        # 提取正负样本的位置号
        pos_inds = torch.nonzero(assigned_gt_inds > 0).reshape(-1)   # 为了返回位置号，在pytorch没有np.where下，借用torch.nonzero()可以返回位置号
        neg_inds = torch.nonzero(assigned_gt_inds == 0).reshape(-1)
    
        return pos_inds, neg_inds
        
 
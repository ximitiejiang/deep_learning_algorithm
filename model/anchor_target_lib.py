#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:07:24 2019

@author: ubuntu
"""
import torch
from model.anchor_assigner_sampler_lib import MaxIouAssigner, PseudoSampler


def get_anchor_target(anchor_list, gt_bboxes_list, gt_labels_list,
                      img_metas_list, assigner_cfg, sampler_cfg, 
                      num_level_anchors, target_means, target_stds):
        """在ssd算法中计算一个batch的多张图片的anchor target，也就是对每个anchor定义他所属的label和坐标：
        其中如果是正样本则label跟对应bbox一致，坐标跟对应bbox也一致；如果是负样本则label
        Input:
            anchor_list: (b, )(s, 4)
            gt_bboxes_list: (b, )(k, 4)
            gt_labels_list: (b, )(k, )
            img_metas_list： (b, )(dict)
        """
        # 初始化
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        
        all_pos_inds_list = []
        all_neg_inds_list = []
        # 循环对每张图分别计算anchor target
        for i in range(len(img_metas_list)):        
            bbox_targets, bbox_weights, labels, label_weights, pos_inds, neg_inds = \
                get_one_img_anchor_target(anchor_list[i],
                                          gt_bboxes_list[i],
                                          gt_labels_list[i],
                                          img_metas_list[i],
                                          assigner_cfg,
                                          sampler_cfg,
                                          target_means,
                                          target_stds)
            # batch图片targets汇总
            all_bbox_targets.append(bbox_targets)   # (b, ) (n_anchor, 4)
            all_bbox_weights.append(bbox_weights)   # (b, ) (n_anchor, 4)
            all_labels.append(labels)                # (b, ) (n_anchor, )
            all_label_weights.append(label_weights)  # (b, ) (n_anchor, )
            all_pos_inds_list.append(pos_inds)       # (b, ) (k, )
            all_neg_inds_list.append(neg_inds)       # (b, ) (j, )
            
        # 对targets数据进行变换，按照特征图个数把每个数据分成6份，把多张图片的同尺寸特征图的数据放一起，统一做loss
        all_bbox_targets = torch.stack(all_bbox_targets, dim=0)   # (b, n_anchor, 4)
        all_bbox_weights = torch.stack(all_bbox_weights, dim=0)   # (b, n_anchor, 4)
        all_labels = torch.stack(all_labels, dim=0)               # (b, n_anchor)
        all_label_weights = torch.stack(all_label_weights, dim=0) # (b, n_anchor)
        
        num_total_pos = sum([inds.numel() for inds in all_pos_inds_list])
        num_total_neg = sum([inds.numel() for inds in all_neg_inds_list])

        return (all_bbox_targets, all_bbox_weights, all_labels, all_label_weights, num_total_pos, num_total_neg)


# %%
def get_one_img_anchor_target(anchors, gt_bboxes, gt_labels, img_metas, 
                              assigner_cfg, sampler_cfg, target_means, target_stds):
    """针对单张图的anchor target计算： 本质就是把正样本的坐标和标签作为target
    提供一种挑选target的思路：先iou筛选出正样本(>0)和负样本(=0)，去掉无关样本(-1)
    然后找到对应正样本anchor, 换算出对应bbox和对应label
    
    args:
        anchors (s, 4)
        gt_bboxes (k, 4)
        gt_labels (k, )
        img_metas (dict)
        assigner_cfg (dict)
        sampler_cfg (dict)
    """
    # 1.指定: 指定每个anchor是正样本还是负样本(通过让anchor跟gt bbox进行iou计算来评价，得到每个anchor的)
    bbox_assigner = MaxIouAssigner(**assigner_cfg.params)
    assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_labels)
    assigned_gt_inds, assigned_gt_labels, ious = assign_result  # (m,), (m,) 表示anchor的身份, anchor对应的标签，[0,1,0,..2], [0,15,...18]
    
    # 2. 采样： 采样一定数量的正负样本, 通常用于预防正负样本不平衡
    #　但在SSD中没有采样只是区分了一下正负样本，所以这里只是一个假采样。(正负样本不平衡是通过最后loss的OHEM完成)
    bbox_sampler = PseudoSampler(**sampler_cfg.params)
    sampling_result = bbox_sampler.sample(*assign_result, anchors, gt_bboxes)
    pos_inds, neg_inds = sampling_result  # (k,), (j,) 表示正/负样本的位置号，比如[7236, 7249, 8103], [0,1,2...8104..]
    
    # 3. 初始化target    
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(len(anchors), dtype=torch.long)       # 借用anchors的device
    label_weights = anchors.new_zeros(len(anchors), dtype=torch.long)# 借用anchors的device
    # 4. 把正样本 bbox坐标转换成delta坐标并填入
    pos_bboxes = anchors[pos_inds]                   # (k,4)获得正样本bbox
    
    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1 # 表示正样本对应的label也就是gt_bbox是第0个还是第1个(已经减1，就从1-n变成0-n-1)
    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]  # (k,4)获得正样本bbox对应的gt bbox坐标
    
    pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means, target_stds)  
    bbox_targets[pos_inds] = pos_bbox_targets
    bbox_weights[pos_inds] = 1
    # 5. 把正样本labels填入
    labels[pos_inds] = gt_labels[pos_assigned_gt_inds] # 获得正样本对应label
    label_weights[pos_inds] = 1  # 这里设置正负样本权重都为=1， 如果有需要可以提高正样本权重
    label_weights[neg_inds] = 1
    
    return bbox_targets, bbox_weights, labels, label_weights, pos_inds, neg_inds


# %%
def bbox2delta(prop, gt, means, stds):
    """把proposal的anchor(k, 4)转化为相对于gt(k,4)的变化dx,dy,dw,dh
    基本逻辑：由前面的卷积网络可以得到预测xmin,ymin,xmax,ymax，并转化成px,py,pw,ph.
    此时存在一种变换dx,dy,dw,dh，可以让预测值变成gx',gy',gw',gh'且该值更接近gx,gy,gw,gh
    所以目标就变成找到dx,dy,dw,dh，寻找的方式就是dx=(gx-px)/pw, dy=(gy-py)/ph, dw=log(gw/pw), dh=log(gh/ph)
    因此卷积网络前向计算每次都得到xmin/ymin/xmax/ymax经过head转换成dx,dy,dw,dh，力图让loss最小使这个变换
    最后测试时head计算得到dx,dy,dw,dh，就可以通过delta2bbox()反过来得到xmin,ymin,xmax,ymax
    """
    # 把xmin,ymin,xmax,ymax转换成x_ctr,y_ctr,w,h
    px = (prop[...,0] + prop[...,2]) * 0.5
    py = (prop[...,1] + prop[...,3]) * 0.5
    pw = (prop[...,2] - prop[...,0]) + 1.0  
    ph = (prop[...,3] - prop[...,1]) + 1.0
    
    gx = (gt[...,0] + gt[...,2]) * 0.5
    gy = (gt[...,1] + gt[...,3]) * 0.5
    gw = (gt[...,2] - gt[...,0]) + 1.0  
    gh = (gt[...,3] - gt[...,1]) + 1.0
    # 计算dx,dy,dw,dh
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1) # (n, 4)
    # 归一化
    means = deltas.new_tensor(means).reshape(1,-1)   # (1,4)
    stds = deltas.new_tensor(stds).reshape(1,-1)      # (1,4)
    deltas = (deltas - means) / stds    # (n,4)-(1,4) / (1,4) -> (n,4) / (1,4) -> (n,4) 
    
    return deltas



def delta2bbox():
    pass   
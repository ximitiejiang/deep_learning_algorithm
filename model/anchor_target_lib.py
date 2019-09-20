#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:07:24 2019

@author: ubuntu
"""
import torch
from model.anchor_assigner_sampler_lib import MaxIouAssigner, PseudoSampler

def bbox2delta(prop, gt):
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
    means = [0,0,0,0]    # (4,)
    std = [0.1,0.1,0.2,0.2]      # (4,)
    deltas = (deltas - means[None, :]) / std[None, :]    # (n,4)-(1,4) / (1,4) -> (n,4) / (1,4) -> (n,4) 
    

def delta2bbox():
    pass   


def get_anchor_target(self, anchor_list, gt_bboxes_list, gt_labels_list,
                      img_metas_list, assigner_cfg, num_level_anchors):
        """在ssd算法中计算一个batch的多张图片的anchor target，也就是对每个anchor定义他所属的label和坐标：
        其中如果是正样本则label跟对应bbox一致，坐标跟对应bbox也一致；如果是负样本则label
        Input:
            anchor_list: (b, )(s, 4)
            gt_bboxes_list: (b, )(k, 4)
            img_metas_list： (b, )(dict)
            gt_labels_list: (b, )(m, )
        """
        # TODO: 放在哪个模块里边比较合适
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        
        all_pos_inds_list = []
        all_neg_inds_list = []
        # 对每张图分别计算anchor target
        for i in range(len(img_metas_list)): 
            # 1.指定: 指定每个anchor是正样本还是负样本(基于跟gt进行iou计算)
            bbox_assigner = MaxIouAssigner(**assigner_cfg)
            assign_result = bbox_assigner.assign(anchor_list[i], gt_bboxes_list[i], gt_labels_list[i])
            # 2.采样： 采样一定数量的正负样本:通常用于预防正负样本不平衡
            #　但在SSD中没有采样只是区分了一下正负样本，所以这里只是一个假采样。(正负样本不平衡是通过最后loss的OHEM完成)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchor_list[i], gt_bboxes_list[i])
            assigned_gt_inds, assigned_gt_labels, ious = assign_result  # (m,), (m,) 表示anchor的身份, anchor对应的标签，[0,1,0,..2], [0,15,...18]
            pos_assigned_gt_inds = assigned_gt_inds[assigned_gt_inds > 0] - 1 #从1～k变换到0~k-1
            pos_inds, neg_inds = sampling_result  # (k,), (j,) 表示正/负样本的位置号，比如[7236, 7249, 8103], [0,1,2...8104..]
            
            # 3. 计算每个anchor的target：基于找到的正样本，得到bbox targets, bbox_weights     
            bbox_targets = torch.zeros_like(anchor_list[i])
            bbox_weights = torch.zeros_like(anchor_list[i])
            labels = torch.zeros(len(anchor_list[i]), dtype=torch.long)
            label_weights = torch.zeros(len(anchor_list[i]), dtype=torch.long)
            
            pos_bboxes = anchor_list[i][pos_inds]                # (k,4)获得正样本bbox
            pos_gt_bboxes = gt_bboxes_list[i][pos_assigned_gt_inds]  # (k,4)获得正样本bbox对应的gt bbox坐标
            pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes)  
            
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1
            # 得到labels, label_weights
            labels[pos_inds] = gt_labels_list[i][pos_assigned_gt_inds]
            label_weights[pos_inds] = 1  # 这里设置正负样本权重=1， 如果有需要可以提高正样本权重
            label_weights[neg_inds] = 1
            
            # batch图片targets汇总
            all_bbox_targets.append(bbox_targets)   # (n_img, ) (n_anchor, 4)
            all_bbox_weights.append(bbox_weights)   # (n_img, ) (n_anchor, 4)
            all_labels.append(labels)                # (n_img, ) (n_anchor, )
            all_label_weights.append(label_weights)  # (n_img, ) (n_anchor, )
            all_pos_inds_list.append(pos_inds)       # (n_img, ) (k, )
            all_neg_inds_list.append(neg_inds)       # (n_img, ) (j, )
            
        # 4. 对targets数据进行变换，按照特征图个数把每个数据分成6份，把多张图片的同尺寸特征图的数据放一起，统一做loss
        all_bbox_targets = torch.stack(all_bbox_targets, dim=0)   # (n_img, n_anchor, 4)
        all_bbox_weights = torch.stack(all_bbox_weights, dim=0)   # (n_img, n_anchor, 4)
        all_labels = torch.stack(all_labels, dim=0)               # (n_img, n_anchor)
        all_label_weights = torch.stack(all_label_weights, dim=0) # (n_img, n_anchor)
        
        num_total_pos = sum([inds.numel() for inds in all_pos_inds_list])
        num_total_neg = sum([inds.numel() for inds in all_neg_inds_list])
        
#        def distribute_to_level(target, num_level_anchors):
#            """把(n_img, n_anchor, 4)或(n_img, n_anchor, )的数据分配到每个level去，
#             变成(n_level,) (n_imgs, n_anchor)或(n_level,) (n_imgs, n_anchor)"""
#            distributed = []
#            start=0
#            for n in num_level_anchors:
#                end = start + n
#                distributed.append(target[:, start:end])
#                start = end
#        
#        distribute = False
#        if distribute:
#            all_bbox_targets = distribute_to_level(all_bbox_targets, num_level_anchors)  # (6,) (4, 5776, 4)
#            all_bbox_weights = distribute_to_level(all_bbox_weights, num_level_anchors)  # (6,) (4, 5776, 4)
#            all_labels = distribute_to_level(all_labels, num_level_anchors)              # (6,) (4, 5776)  
#            all_label_weights = distribute_to_level(all_label_weights, num_level_anchors)# (6,) (4, 5776)

        return (all_bbox_targets, all_bbox_weights, all_labels, all_label_weights, num_total_pos, num_total_neg)
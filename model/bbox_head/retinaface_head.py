#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:36:18 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
from model.anchor_generator_lib import AnchorGenerator
from model.loss_lib import SmoothL1Loss, CrossEntropyLoss

class ClassHead(nn.Module):
    """分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes=2):
        super.__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * num_classes, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape(0), -1, self.num_classes)
        return out
        

class BboxHead(nn.Module):
    """bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super.__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape(0), -1, 4)
        return out        

    
class LandmarkHead(nn.Module):
    """landmark回归模块"""
    def __init__(self, in_channels, num_anchors, num_points):
        super.__init__()
        self.num_points = num_points
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * num_points, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape(0), -1, self.num_points)
        return out


class RetinaFaceHead(nn.Module):
    
    def __init__(self, 
                 in_channels=(64, 64, 64), 
                 num_anchors=2,
                 num_classes=2,
                 num_landmarks=10):
        super().__init__()
        # 定义分类回归头
        self.class_head = nn.ModuleList()
        for in_channel in in_channels:
            self.class_head.append(ClassHead(in_channel, num_anchors, num_classes))
            
        self.bbox_head = nn.ModuleList()
        for in_channel in in_channels:
            self.bbox_head.append(BboxHead(in_channel, num_anchors))
        
        self.landmark_head = nn.ModuleList()
        for in_channel in in_channels:
            self.landmark_head.append(LandmarkHead(in_channel, num_anchors, num_landmarks))
        
        # 定义损失函数
        self.loss_cls_fn = CrossEntropyLoss()
        self.loss_bbox_fn = SmoothL1Loss(beta=0.2)
        self.loss_ldmk_fn = SmoothL1Loss(beta=0.2)
        
        # 定义anchors
        base_sizes
        scales
        ratios
        centers
        self.anchor_generators = []
        for i in range(len(in_channels)):
            self.anchor_generators.append(AnchorGenerator(
                    base_sizes[i], scales[i], ratios[i], ctr=centers[i], scale_major=False))
            
    
    def init_weights(self):
        pass
    
    
    def forward(self, x):
        cls_scores = [self.class_head[i](x[i]) for i in range(len(x))] #(3,) (b,-1,2)
        cls_scores = torch.cat(cls_scores, dim=1) # (b,-1,2)
        
        bbox_preds = [self.bbox_head[i](x[i]) for i in range(len(x))]
        bbox_preds = torch.cat(bbox_preds, dim=1) # (b,-1,4)
        
        ldmk_preds = [self.landmark_head[i](x[i]) for i in range(len(x))]
        ldmk_preds = torch.cat(ldmk_preds, dim=1) # (b,-1,10)
        
        return tuple(cls_scores, bbox_preds, ldmk_preds)
    
    
    def get_losses(self, cls_scores, bbox_preds, ldmk_preds, 
                   gt_bboxes, gt_labels, gt_landmarks, img_metas, cfg):
        """计算损失，通过anchors把3组gt数据转化成target数据，然后分别跟3组预测数据计算损失
        args:
            cls_scores: (b,-1,2)
            bbox_preds: (b,-1,4)
            ldmk_preds: (b,-1,10)
            gt_bboxes: (b,) (n,4)
            gt_labels: (b,) (n,)
            gt_landmarks: (b,) (n,5,2)
        """
        num_imgs = len(img_metas)
        # 准备anchors
        all_anchors = []
        for i in range():
            all_anchors.append(self.anchor_generators[i].grid_anchors())
        num_anchors = [len(anchor) for anchor in all_anchors]
        all_anchors = torch.cat(all_anchors, dim=0)
        all_anchors = [all_anchors for _ in range(len(img_metas))]
        
        # 开始计算target
        for i in range(num_imgs):
            targets = get_anchor_target(all_anchors, gt_bboxes, gt_labels, gt_landmarks, img_metas)
        cls_scores_t, bbox_preds_t, ldmk_preds_t = targets
        
        # 计算损失
        loss_cls
        loss_reg
        loss_ldmk
        
    def get_anchor_target(anchors, gt_bboxes, gt_labels, img_metas, assigner_cfg, sampler_cfg, num_level_anchors, target_means, target_stds):
        """通过anchors从gt数据中获取target"""
        bbox_t, bbox_w, labels_t, labels_w, pos_inds, neg_inds = map(single_target, )
        
    
    def get_bboxes(self):
        pass
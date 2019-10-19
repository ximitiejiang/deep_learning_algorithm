#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:36:18 2019

@author: ubuntu
"""
import torch
import torch.nn as nn
from math import ceil
from functools import partial
from utils.init_weights import xavier_init
from model.anchor_generator_lib import AnchorGenerator
from model.loss_lib import SmoothL1Loss, CrossEntropyLoss, SigmoidBinaryCrossEntropyLoss
from model.get_target_lib import get_anchor_target

class ClassHead(nn.Module):
    """分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes=2):
        super.__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * num_classes, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, self.num_classes)
        return out
        

class BboxHead(nn.Module):
    """bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super.__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 4)
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
        out = out.view(out.shape[0], -1, self.num_points)
        return out


class RetinaFaceHead(nn.Module):
    
    def __init__(self,
                 input_size=(640, 640),
                 in_channels=(64, 64, 64), 
                 num_classes=2,
                 num_landmarks=10,
                 base_sizes=(16, 64, 256),
                 strides=(8, 16, 32),
                 scales=(1, 2),
                 ratios=(1),
                 means=(0.,0.,0.,0.),
                 stds=(0.1,0.1,0.1,0.1)):
        super().__init__()
        
        self.means = means
        self.stds = stds
        self.featmap_sizes = [[ceil(input_size[0]/stride), ceil(input_size[1]/stride)] for stride in strides]
        num_anchors = len(scales) * len(ratios)
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
        self.loss_bbox_fn = SmoothL1Loss()
        self.loss_ldmk_fn = SigmoidBinaryCrossEntropyLoss()
        
        # 定义anchors
        self.anchor_generators = []
        for i in range(len(in_channels)):
            ctr = ((strides[i] - 1.) / 2, (strides[i] - 1.) / 2)
            self.anchor_generators.append(AnchorGenerator(
                    base_sizes[i], scales, ratios, ctr=ctr, scale_major=False))
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform", bias=0)
    
    
    def forward(self, x):
        cls_scores = [self.class_head[i](x[i]) for i in range(len(x))] #(3,) (b,-1,2)
        cls_scores = torch.cat(cls_scores, dim=1) # (b,-1,2)
        
        bbox_preds = [self.bbox_head[i](x[i]) for i in range(len(x))]
        bbox_preds = torch.cat(bbox_preds, dim=1) # (b,-1,4)
        
        ldmk_preds = [self.landmark_head[i](x[i]) for i in range(len(x))]
        ldmk_preds = torch.cat(ldmk_preds, dim=1) # (b,-1,10)
        
        return tuple(cls_scores, bbox_preds, ldmk_preds)
    
    
    def get_losses(self, cls_scores, bbox_preds, ldmk_preds, 
                   gt_bboxes, gt_labels, gt_landmarks, cfg):
        """计算损失，通过anchors把3组gt数据转化成target数据，然后分别跟3组预测数据计算损失
        args:
            cls_scores: (b,-1,2)
            bbox_preds: (b,-1,4)
            ldmk_preds: (b,-1,10)
            gt_bboxes: (b,) (n,4)
            gt_labels: (b,) (n,)
            gt_landmarks: (b,) (n,5,2)
        """
        num_imgs = len(gt_labels)
        # 准备anchors
        all_anchors = []
        for i in range(len(self.featmap_sizes)):
            device = cls_scores.device
            all_anchors.append(self.anchor_generators[i].grid_anchors(
                    self.featmap_sizes[i], self.strides[i], device=device))   
#        num_anchors = [len(anchor) for anchor in all_anchors]
        all_anchors = torch.cat(all_anchors, dim=0)
        all_anchors = [all_anchors for _ in range(len(num_imgs))]
        # 开始计算target
        target_result = get_anchor_target(all_anchors, gt_bboxes, gt_labels, gt_landmarks,
                                          cfg.assigner, cfg.sampler, 
                                          self.means, self.stds)
        bboxes_t, _, labels_t, _, ldmk_t, *_ = target_result
        
        # 计算损失
        loss_cls = list(map(self.loss_cls_fn, cls_scores, labels_t))
        loss_bbox = list(map(self.loss_bbox_fn, bbox_preds, bboxes_t))
        loss_ldmk = list(map(self.loss_ldmk_fn, ))
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_ldmk=loss_ldmk)
        
    
    def get_bboxes(self):
        pass
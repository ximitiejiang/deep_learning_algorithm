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
import torch.nn.functional as F
from utils.init_weights import xavier_init
from model.anchor_generator_lib import AnchorGenerator
from model.loss_lib import SmoothL1Loss, CrossEntropyLoss
from model.get_target_lib import get_anchor_target
from model.bbox_head.ssd_head import ohem
from model.bbox_regression_lib import delta2bbox, delta2landmark
from model.nms_lib import nms_wrapper

class ClassHead(nn.Module):
    """分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * num_classes, 1, stride=1, padding=0)  # 无论输入多大
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, self.num_classes)
        return out
        

class BboxHead(nn.Module):
    """bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, 1, stride=1, padding=0)
    
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 4)
        return out        

    
class LandmarkHead(nn.Module):
    """landmark回归模块"""
    def __init__(self, in_channels, num_anchors, num_points):
        super().__init__()
        self.num_points = num_points
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2 * num_points, 1, stride=1, padding=0)
    
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
                 num_landmarks=5,
                 base_sizes=(16, 64, 256),
                 strides=(8, 16, 32),
                 scales=(1, 2),
                 ratios=(1),
                 target_means=(0.,0.,0.,0.),
                 target_stds=(0.1,0.1,0.1,0.1),
                 neg_pos_ratio=3):
        super().__init__()
        self.strides = strides
        self.target_means= target_means
        self.target_stds = target_stds
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes

        scales = [scales] if isinstance(scales, int) else scales
        ratios = [ratios] if isinstance(ratios, int) else ratios  # 如果输入的是单个数，比如2，或(2)，则需要转换成list
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
        self.loss_ldmk_fn = SmoothL1Loss()
        
        # 定义anchors: 基于retinaface的
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
        self.featmap_sizes = [feat.shape[2:] for feat in x]
        
        cls_scores = [self.class_head[i](x[i]) for i in range(len(x))] #(3,) (b,-1,2)
        cls_scores = torch.cat(cls_scores, dim=1) # (b,-1,2)
        
        bbox_preds = [self.bbox_head[i](x[i]) for i in range(len(x))]
        bbox_preds = torch.cat(bbox_preds, dim=1) # (b,-1,4)
        
        ldmk_preds = [self.landmark_head[i](x[i]) for i in range(len(x))]
        ldmk_preds = torch.cat(ldmk_preds, dim=1) # (b,-1,10)
        
        return cls_scores, bbox_preds, ldmk_preds
    
    
    def get_losses(self, cls_scores, bbox_preds, ldmk_preds, 
                   gt_bboxes, gt_labels, gt_landmarks, cfg, **kwargs):
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
        all_anchors = torch.cat(all_anchors, dim=0)
        all_anchors = [all_anchors for _ in range(num_imgs)]
        # 开始计算target
        target_result = get_anchor_target(all_anchors, gt_bboxes, gt_labels, gt_landmarks,
                                          cfg.assigner, cfg.sampler, 
                                          self.target_means, self.target_stds)
        bboxes_t, bboxes_w, labels_t, labels_w, ldmk_t, ldmk_w, num_pos, num_neg = target_result
        
        # 计算分类损失
        loss_cls = list(map(self.loss_cls_fn, cls_scores, labels_t))   # (b,-1,2) and (b,-1)
        loss_cls = [loss_cls[i] * labels_w[i].float() for i in range(len(loss_cls))]
        pfunc = partial(ohem, neg_pos_ratio=self.neg_pos_ratio, avg_factor=num_pos)
        loss_cls = list(map(pfunc, loss_cls, labels_t))
        # 计算回归损失: 注意回归损失增加了一个倍数2的权重
        pfunc = partial(self.loss_bbox_fn, avg_factor=num_pos)         
        loss_bbox = 2 * list(map(pfunc, bbox_preds, bboxes_t, bboxes_w))    # (b,-1,4) and (b,-1,4)
        # 计算关键点损失
        pfunc = partial(self.loss_ldmk_fn, avg_factor=num_pos)
        loss_ldmk = list(map(pfunc, ldmk_preds, ldmk_t, ldmk_w))         # (b,-1,10) and (b,-1,10)
        
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_ldmk=loss_ldmk)
        
    
    def get_bboxes(self, cls_scores, bbox_preds, ldmk_preds, img_metas, cfg, *kwargs):
        """在测试时基于前向计算结果，计算bbox预测类别和预测坐标，此时前向计算后不需要算loss，直接计算bbox的预测
        注意：这部分当前只支持单图计算，但数据为了能通过前向计算都包装成batch size=1
        Args:
            cls_scores: (1,-1,2)
            bbox_preds: (1,-1,4)
            ldmk_preds: (1,-1,10)
            img_metas:(1,)
            cfg:
        """
        # 拆包装
        if cls_scores.shape[0] == 1:
            cls_scores = cls_scores[0] # (-1,2)
            bbox_preds = bbox_preds[0] # (-1,4)
            ldmk_preds = ldmk_preds[0] # (-1,10)
            img_metas = img_metas[0]   # dict
        else:
            raise ValueError('only support batch size=1 prediction.')
        # 准备anchors
        img_size = img_metas['pad_shape']
        anchors = []
        for i in range(len(self.featmap_sizes)):
            device = cls_scores.device
            anchors.append(self.anchor_generators[i].grid_anchors(
                    self.featmap_sizes[i], self.strides[i], device=device))   
        anchors = torch.cat(anchors, dim=0)     
        # 计算单张图的bbox预测
        scale_factor = img_metas['scale_factor']
        
        cls_scores = F.softmax(cls_scores, dim=1) # 概率化
        bbox_preds = delta2bbox(anchors, bbox_preds, self.target_means, self.target_stds, img_size) # 坐标化
        ldmk_preds = delta2landmark(anchors, ldmk_preds, self.target_means, self.target_stds)
        bboxes_preds = bbox_preds / bbox_preds.new_tensor(scale_factor[:4])  # 相对原图的尺寸
        # nms
        bboxes, labels, ldmks = nms_wrapper(bboxes_preds, cls_scores, ldmk_preds, **cfg.nms) # (n_cls,)(m,5),  (n_cls,)(m,),  (n_cls,)(m,5,2) 

        return dict(bboxes=bboxes, labels=labels, ldmks=ldmks) # (n_cls,)(m,5)   (n_cls,)(m,)  (n_cls,)(m,5,2)    




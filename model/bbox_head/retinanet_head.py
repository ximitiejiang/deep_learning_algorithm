#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:09:48 2019

@author: ubuntu
"""
import torch.nn as nn
import torch
from functools import partial

from utils.init_weights import normal_init, bias_init_with_prob
from utils.tools import one_hot_encode
from model.get_target_lib import get_anchor_target
from model.anchor_generator_lib import AnchorGenerator
from model.loss_lib import SigmoidFocalLoss, SmoothL1Loss
"""
header=dict(
        type='retina_head',
        params=dict(
                input_size=300,
                num_classes=21,
                in_channels=(512, 1024, 512, 256, 256, 256),
                num_anchors=(4, 6, 6, 6, 4, 4),
                anchor_strides=(8, 16, 32, 64, 100, 300),
                target_means=(.0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2)))
"""
#def conv3x3(in_channels, out_channels, stride, padding, bias):
#    
#    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
#                         nn.ReLU(inplace=True))
    
class ClassHead(nn.Module):
    """针对单层特征的分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True))
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.cls_convs(x)
        out = self.cls_head(x)
        out = out.permute(0, 2, 3, 1).contiguous()
#        out = out.view(out.shape[0], -1, self.num_classes)  
        out = out.view(int(out.size(0)), int(-1), int(self.num_classes))
        return out
    
    def init_weights(self):
        for m in self.cls_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_head, std=0.01, bias=bias_cls)
        

class BboxHead(nn.Module):
    """针对单层特征的bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.reg_convs = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, in_channels, 3, 1, 1, True),
                                       nn.ReLU(inplace=True))
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.reg_convs(x)
        out = self.reg_head(x)
        out = out.permute(0, 2, 3, 1).contiguous()
#        out = out.view(out.shape[0], -1, 4)
        out = out.view(int(out.size(0)), int(-1), int(4))
        return out     
    
    def init_weights(self):
        for m in self.reg_convs:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reg_head, std=0.01, bias=bias_cls)

    
class RetinaNetHead(nn.Module):
    """retina head"""
    def __init__(self, 
                 input_size=(1333, 800),
                 num_classes=21,
                 in_channels=(256, 256, 256, 256, 256),
                 base_scale=4,
                 ratios = [1/2, 1, 2],
                 anchor_strides=(8, 16, 32, 64, 128),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 alpha=0.25,
                 gamma=2,
                 **kwargs):
        
        super().__init__()        
        self.num_classes = num_classes
        self.anchor_strides = anchor_strides
        self.target_means = target_means
        self.target_stds = target_stds
        # 参数
        """retinanet生成anchor的逻辑：3个核心参数的定义过程
        base_size = [8, 16, 32, 64, 128] 采用的就是strides
        scales = 4*[2**(i/3) for i in range(3)] 采用的是在基础比例[1, 1.2, 1.5]的基础上乘以4, 其中基础比例的定义感觉是经验，乘以4感觉是为了匹配原图
        定义了一个octave_base_scale=4，然后定义了sctave_scales=[1, 1.2599, 1.5874]"""
        scales =  [base_scale * 2**(i / 3) for i in range(3)]
        base_sizes = anchor_strides
        # 创建anchor生成器
        self.anchor_generators = []
        for i in range(len(anchor_strides)):
            anchor_generator = AnchorGenerator(base_sizes[i], scales, 
                                               ratios, scale_major=False) 
            self.anchor_generators.append(anchor_generator)
        # 创建分类回归头
        num_anchors = len(ratios) * len(scales)
        self.cls_head = ClassHead(in_channels[0], num_anchors, num_classes-1)
        self.reg_head = BboxHead(in_channels[0], num_anchors)

        # 创建损失函数
        self.loss_cls = SigmoidFocalLoss(alpha=alpha, gamma=gamma)
        self.loss_bbox = SmoothL1Loss()
    
    def init_weights(self):
        self.cls_head.init_weights()
        self.reg_head.init_weights()
    
    def forward(self, x):
        self.featmap_sizes = [feat.shape[2:] for feat in x] 
        cls_scores = []
        bbox_preds = []
        for feat in x:
            cls_scores.append(self.cls_head(feat))
            bbox_preds.append(self.reg_head(feat))
        return cls_scores, bbox_preds  # 这是模型最终输出，最好不用dict，避免跟onnx inference冲突
    
    def get_losses(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, cfg, **kwargs):
        """retinanet
        cls_scores(5,) (b, -1, 20)
        bbox_preds(5,) (b, -1, 4)
        gt_bboxes(b, )
        gt_labels(b, )
        """
        num_imgs = len(gt_labels)
        multi_layer_anchors = []
        for i in range(len(self.featmap_sizes)):
            device = cls_scores[0].device
            anchors = self.anchor_generators[i].grid_anchors(
                self.featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_layer_anchors.append(anchors)  # (6,)(k, 4)
        multi_layer_anchors = torch.cat(multi_layer_anchors, dim=0)   # 堆叠(5,)(m, 4)->(652239, 4)    
        anchor_list = [multi_layer_anchors for _ in range(num_imgs)]  # 复制batch份
        # 计算target: None表示gt_landmarks=None
        target_result = get_anchor_target(anchor_list, gt_bboxes, gt_labels, None,
                                          cfg.assigner, cfg.sampler,
                                          self.target_means, self.target_stds)
        # 解析target
        bboxes_t, bboxes_w, labels_t, labels_w, _, _, num_pos, num_neg = target_result  # (b,-1,4)x2, (b,-1)x2
        """retinanet的变化：只取正样本数量作为total_sample"""
        
        """retinanet的变化：labels需要转换成独热编码方式输入focal loss"""
        labels_t = one_hot_encode(labels_t, self.num_classes-1)  # (b, 728973)每张图的5张特征图放在一起get_target了
        labels_w = labels_w.view(-1, 1).expand(labels_w.size(0), self.num_classes-1)
        """retinanet的变化：计算损失时是把1个batch的比如4张图的某一特征层的labels, weights放在一起算，即(b, -1, 20)reshape成(-1, 20)
        但我这里调整了，改成一张图的5个featmap放在一起算，4张图就map4次
        """
        # cls分类损失
        pfunc = partial(self.loss_cls, avg_factor=num_pos)
        loss_cls = list(map(pfunc, cls_scores, labels_t, labels_w))
#        loss_cls = [loss_cls[i] * labels_w[i].float() for i in range(len(loss_cls))]  # (b,)(8732,)
        # cls loss的ohem
#        pfunc = partial(ohem, neg_pos_ratio=self.neg_pos_ratio, avg_factor=num_pos)
#        loss_cls = list(map(pfunc, loss_cls, labels_t))   # (b,)
        # bbox回归损失
        pfunc = partial(self.loss_bbox, avg_factor=num_pos)
        loss_bbox = list(map(pfunc, bbox_preds, bboxes_t, bboxes_w))  # (b,)
        return dict(loss_cls = loss_cls, loss_bbox = loss_bbox)  # {(b,), (b,)} 每张图对应一个分类损失值和一个回归损失值。        


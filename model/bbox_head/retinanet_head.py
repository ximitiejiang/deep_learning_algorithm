#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:09:48 2019

@author: ubuntu
"""
import torch.nn as nn

from model.anchor_generator_lib import AnchorGenerator
from model.loss_lib import FocalLoss
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
def conv3x3(in_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels),
                         nn.ReLU())
    
class ClassHead(nn.Module):
    """针对单层特征的分类模块"""
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv3x3 = nn.Conv2d(in_channels, num_anchors * num_classes, 3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv3x3(x)
        out = out.permute(0, 2, 3, 1).contiguous()
#        out = out.view(out.shape[0], -1, self.num_classes)  
        out = out.view(int(out.size(0)), int(-1), int(self.num_classes))
        return out
        

class BboxHead(nn.Module):
    """针对单层特征的bbox回归模块"""
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, num_anchors * 4, 3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv3x3(x)
        out = out.permute(0, 2, 3, 1).contiguous()
#        out = out.view(out.shape[0], -1, 4)
        out = out.view(int(out.size(0)), int(-1), int(4))
        return out     

    

class RetinaNetHead(nn.Module):
    """retina head"""
    def __init__(self, 
                 input_size=(1333, 800),
                 num_classes=21,
                 in_channels=256,
                 base_scale=4,
                 ratios = [1/2, 1, 2],
                 anchor_strides=(8, 16, 32, 64, 128),
                 loss_cls_cfg=None,
                 loss_reg_cfg=None,
                 **kwargs):
        
        super().__init__()        
        
        # 参数
        """retinanet生成anchor的逻辑：3个核心参数的定义过程
        base_size = [8, 16, 32, 64, 128] 采用的就是strides
        scales = 4*[2**(i/3) for i in range(3)] 采用的是在基础比例[1, 1.2, 1.5]的基础上乘以4, 其中基础比例的定义感觉是经验，乘以4感觉是为了匹配原图
        定义了一个octave_base_scale=4，然后定义了sctave_scales=[1, 1.2599, 1.5874]"""
        scales = base_scale * [2**(i / 3) for i in range(3)]
        base_sizes = anchor_strides
        # 创建anchor生成器
        self.anchor_generators = []
        for i in range(len(in_channels)):
            anchor_generator = AnchorGenerator(base_sizes[i], scales[i], 
                                               ratios[i], scale_major=False) 
            self.anchor_generators.append(anchor_generator)
        
        # 创建分类回归头
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in range(4):
            self.cls_convs.append(conv3x3())
            self.reg_convs.append(conv3x3())
        num_anchors = len(ratios) * len()
        self.cls_head = ClassHead(in_channels, )
        self.reg_head = BboxHead()

        # 创建损失函数
        self.loss_cls = CrossEntropyLoss()
        self.loss_bbox = SmoothL1Loss()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:37:59 2019

@author: ubuntu
"""

import numpy as np
import torch.nn as nn
from utils.anchor_generator import AnchorGenerator
from model.weight_init import kaiming_normal_init
from utils.registry_build import registered
from .ssd_head import SSDHead

@registered.register_module
class M2detHead(SSDHead):
    """M2detHead主要完成3件事：生成anchors, 处理feat maps，计算loss
    1. anchors的生成：输入anchor跟原图的尺寸比例
       先基于anchor的尺寸比例乘以img尺寸得到min_size, max_size，然后基于strides
       计算anchor在cell的中心点坐标ctx,cty, 然后定义scales=[1,sqrt(max_size/min_size)]
       以及定义ratio=[1,2,1/2,3,1/3], 取scale=1的5种ratio，然后区scale=sqrt(max_size/min_size)的1种ratio=1
    """
    def __init__(self,
                 input_size = 512,      # 相同保留
                 planes = 256,           # 代表tums最终输出层数
                 num_levels = 8,
                 num_classes = 81,
                 anchor_strides = [8, 16, 32, 64, 128, 256],  # 代表特征图的缩放比例，也就是每个特征图上cell对应的原图大小，也就是原图所要布置anchor的尺寸空间(可以此计算ctx,cty)
                 size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],  # 代表anchors尺寸跟img的比例, 前6个数是min_size比例，后6个数是max_size比例，可以此计算anchor最小最大尺寸
                 size_featmaps = [(64,64), (32,32), (16,16), (8,8), (4,4), (2,2)],
                 anchor_ratio_range = ([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]),# 这里的2代表了2和1/2, 而3代表了3和1/3，可以此计算每个cell的anchor个数：2个方框+4个ratio=6个
                 target_means=(.0, .0, .0, .0),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):  
        super().__init__(input_size=input_size, 
                         num_classes=num_classes,            
                         anchor_strides=anchor_strides,
                         anchor_ratios=anchor_ratio_range,
                         target_means=target_means,
                         target_stds=target_stds,
                         **kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self.featmap_sizes = size_featmaps
        self.anchor_strides = anchor_strides
        self.anchor_ratios = anchor_ratio_range
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        
        # create m2det head layers
        reg_convs = []
        cls_convs = []
        for i in range(len(size_featmaps)):
            reg_convs.append(
                nn.Conv2d(
                    planes * num_levels,
                    4 * 6,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    planes * num_levels,
                    num_classes * 6,
                    kernel_size=3,
                    stride=1,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        
        # generate anchors
        if input_size == 512:
            min_ratios = size_pattern[:-1]
            max_ratios = size_pattern[1:]
            
            min_sizes = [min_ratios[i] * input_size for i in range(len(min_ratios))]
            max_sizes = [max_ratios[i] * input_size for i in range(len(max_ratios))]
        
        self.anchor_generators = []
        for k in range(len(self.anchor_strides)):
            base_size = min_sizes[k]
            stride = self.anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]  # 以base_size为第一个anchor，即scale=1
            ratios = [1.]
            for r in self.anchor_ratios[k]:
                ratios += [1 / r, r]
            # scales (2,) and ratios (5,), if scale_major=False, (1,5)*(2,1)->(2,5)*(2,5)->(2,5)
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            
            # for m2det, there are 6 anchors for each cells, no matter in any featmap cell
            anchor_generator.base_anchors = anchor_generator.base_anchors[:6]  # 取前6个anchors(对应scale=1的5种ratios + scale=sqrt(max_size/min_size)的1种ratio)
#            indices = list(range(len(ratios)))
#            indices.insert(1, len(indices))
#            anchor_generator.base_anchors = torch.index_select(
#                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)
    
    def init_weights(self):
        """use m2det init method"""
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        kaiming_normal_init(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0   
        
        self.cls_convs.apply(weights_init)
        self.reg_convs.apply(weights_init)
    

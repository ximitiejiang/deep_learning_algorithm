#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:48:09 2019

@author: ubuntu
"""
import torch.nn as nn
from utils.init_weights import common_init_weights

class FCN8sHead(nn.Module):
    """FCN分割模型头"""
    def __init__(self, 
                 last_channels=512, 
                 num_classes=21, 
                 n_layers=3):
        super().__init__()
        # 卷积分割头
        self.conv_seg = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1))
        # 水平降维头：类似于FPN的literals(1x1)用于统一各个特征图的通道数
        self.literals = nn.ModuleList(nn.Conv2d(256, num_classes, 1),
                                      nn.Conv2d(512, num_classes, 1))
    
    def forward(self, x):
        """从fcnvgg过来x为(3,)
        """
        # 先计算literals进行通道数统一到21
        l_outs = []
        l_outs[0] = x[0]
        for i in range(len(x), 1, -1): # 从高语义层往低语义层
            l_outs[i - 1] = self.literals[i](x[i + 1])
    
    def init_weights(self):
        common_init_weights(self, pretrained=self.pretrained)
        
        
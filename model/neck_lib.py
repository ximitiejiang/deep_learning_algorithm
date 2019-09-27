#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:17:58 2019

@author: ubuntu
"""
import torch.nn as nn
import torch
from model.activation_lib import activation_dict

def conv_bn_relu(in_channels, out_channels, kernel_size, 
                 with_bn=False, activation='relu', with_maxpool=True, 
                 stride=1, padding=1, ceil_mode=False):
    """卷积1x1 (基于vgg的3x3卷积集成模块)：
    - 可包含n个卷积(2-3个)，但卷积的通道数默认在第一个卷积变化，而中间卷积不变，即默认s=1,p=1(这种设置尺寸能保证尺寸不变)。
      所以只由第一个卷积做通道数修改，只由最后一个池化做尺寸修改。
    - 可包含n个bn
    - 可包含n个激活函数
    - 可包含一个maxpool: 默认maxpool的尺寸为2x2，stride=2，即默认特征输出尺寸缩减1/2
    输出：
        layer(list)
    """
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding))
    # bn
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    # activation
    activation_class = activation_dict[activation] 
    layers.append(activation_class(inplace=True))
    in_channels = out_channels
    # maxpool
    if with_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers

"""
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
"""

class FPN(nn.Module):
    """FPN的结构分4部分：
    1. 一组1x1：叫lateral_conv
    2. 累加操作：
    3. 一组3x3：叫fpn_conv
    4. 输出：为了增加输出分组，可能增加maxpool下采样
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 
                 ):
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range():
            lateral_conv = conv_bn_relu(in_channels, out_channels, 1, True, 'relu', False, 1, 1)
            fpn_conv = conv_bn_relu(out_channels, out_channels, 3, True, 'relu', False, 1, 1)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        
        if add_extra_convs:
            for i in range(extra_levels):
                
            
    def forward(self):
        pass
    
    
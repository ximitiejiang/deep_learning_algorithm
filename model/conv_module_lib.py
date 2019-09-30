#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 08:05:15 2019

@author: ubuntu
"""
import torch.nn as nn
from model.activation_lib import activation_dict

def conv_norm_acti(in_channels, out_channels, kernel_size, 
                  norm='bn', activation='relu', 
                  stride=1, padding=1):
    """卷积集成模块，包含卷积层、归一化层、激活层：
    - conv:  可自定义卷积核尺寸
    - norm: 可选择batchnorm, groupnorm
    - act: 可选择relu, elu, leaky_relu
    输出：
        sequential(module list)
    """
    layers = []

    # conv
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding))
    # norm
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'gn':
        layers.append(nn.GroupNorm(out_channels))
    # activation
    activation_class = activation_dict[activation] 
    layers.append(activation_class(inplace=True))

    return nn.Sequential(*layers)
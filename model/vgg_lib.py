#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:14:53 2019

@author: ubuntu
"""

import torch.nn as nn
from utils.init_weights import common_init_weights
from model.activation_lib import activation_dict


def vgg3x3(num_convs, in_channels, out_channels, with_bn=False, activation='relu', with_maxpool=True, 
            stride=1, padding=1, ceil_mode=False):
    """vgg的3x3卷积集成模块：
    - 可包含n个卷积(2-3个)，但卷积的通道数默认在第一个卷积变化，而中间卷积不变，即默认s=1,p=1(这种设置尺寸能保证尺寸不变)。
      所以只由第一个卷积做通道数修改，只由最后一个池化做尺寸修改。
    - 可包含n个bn
    - 可包含n个激活函数
    - 可包含一个maxpool: 默认maxpool的尺寸为2x2，stride=2，即默认特征输出尺寸缩减1/2
    输出：
        layer(list)
    """
    layers = []
    for i in range(num_convs):
        # conv3x3
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
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


class VGG(nn.Module):
    """经典vgg网络：以2个3x3和3个3x3为基本模块搭建而成。
    """
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    
    def __init__(self, depth, pretrained=None, classify_classes=None):
        super().__init__()
        self.pretrained = pretrained
        blocks = self.arch_settings[depth]
        
        layers = []
        in_channels = 3
        for i, block in enumerate(blocks):
            out_channels = 64 * pow(2, i) if i < 4 else 512
            layers.extend(vgg3x3(block, in_channels, out_channels, 
                                 with_bn=False, activation='relu', 
                                 with_maxpool=True))  # 注意要用extend组成一个list
            in_channels = out_channels
        # 组成特征层    
        self.features = nn.Sequential(*layers)
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        # 分类层
        if classify_classes is not None:
            self.classifier = nn.Sequential(
                    nn.Linear(out_channels*7*7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, classify_classes))
        
        self.init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        common_init_weights(self, pretrained=self.pretrained)
        

if __name__ == '__main__':
    import torch
    import torchvision
    import numpy as np
    
    name = 'my'
    
    if name == 'ori':
        model = torchvision.models.vgg16()
    
    if name == 'my':
        vgg = VGG(depth=16, 
                  pretrained = '/home/ubuntu/MyWeights/vgg16-397923af.pth',
                  classify_classes=10)
        img = np.random.randn(8,3,300,300)
        img = torch.Tensor(img)
        output = vgg(img)
    
    
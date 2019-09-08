#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:35:36 2019

@author: ubuntu
"""
import torch.nn as nn
from utils.init_weights import init_weights

activation_dict = {'sigmoid':nn.Sigmoid,
                   'relu':nn.ReLU,
                   'elu':nn.ELU,
                   'leaky_relu':nn.LeakyReLU}

def conv3x3(in_channels, out_channels, activation='relu', with_bn=True, stride=1, padding=1):
    """标准化基础conv3x3: 该标准conv默认不改变特征图尺寸(s=1,p=1)"""
    module = [nn.Conv2d(in_channels, out_channels, 3, stride, padding)]
    if with_bn:
        module.append(nn.BatchNorm2d(out_channels))
    activation_class = activation_dict[activation]
    module.append(activation_class(inplace=True))
    return nn.Sequential(*module)


class AlexNet8(nn.Module):
    """修改自alexnet的一个卷积模型，一共8层(5c + 3l)可学习层。
    用于cifar10: RGB, 32*32, 如果用在大图上可微调全连接输入应该就可以，
    """
    def __init__(self, n_classes):
        super().__init__()
        activation = 'elu'
        self.features = nn.Sequential(
                conv3x3(3, 64, activation, True, 1, 1),
#                conv3x3(64, 64, activation, True, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2), # w=(32-2)/2 +1 =16 
                
                conv3x3(64, 192, activation, True, 1, 1),
#                conv3x3(192, 192, activation, True, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2), #w=(16-2)/2+1=8
                
                conv3x3(192, 384, activation, True, 1, 1),
                conv3x3(384, 256, activation, True, 1, 1),
                conv3x3(256, 256, activation, True, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2)) # w=(8-2)/2+1=4
        
        self.classifier = nn.Sequential(
                nn.Linear(256*4*4, 4096),  # 256*4*4 -> 4096
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, n_classes))
        
        self._init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)           # 展平操作，类似flatten层的操作
        x = self.classifier(x)
        return x
    
    def _init_weights(self):
        init_weights(self, pretrained=None)
        
        
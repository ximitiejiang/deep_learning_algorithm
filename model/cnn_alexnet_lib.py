#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:35:36 2019

@author: ubuntu
"""
import torch.nn as nn
from utils.init_weights import init_weights

class AlexNet8(nn.Module):
    """修改自alexnet的一个卷积模型，一共8层可学习层。
    用于cifar10: RGB, 32*32, 如果用在大图上可微调全连接输入应该就可以，
    """
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # w=(32-3+2)/1 +1=32
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), # w=(32-2)/2 +1 =16 
                
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),# w=(16-3+2)/1+1=16
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), #w=(16-2)/2+1=8
                
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), # w=(8-3+2)/1+1=8
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # w=8
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # w=8
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
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
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:14:07 2019

@author: ubuntu
"""
import torch
import torch.nn as nn

class ClassHead(nn.Module):
    def __init__(self,inchannels,num_anchors):
        super().__init__()
#        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*2, kernel_size=(3,3), stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*2, kernel_size=(1,1), stride=1, padding=0)
        
    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape(0), -1, 2)    # 我操，以为一模一样的两个程序，就因为小括号浪费我一个上午的时间。


class ClassHead2(nn.Module):
    def __init__(self,inchannels,num_anchors):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)
    
    
if __name__ =='__main__':
    m = ClassHead(64, 4)
    img = torch.randn(4,64,32,32)
    out = m(img)
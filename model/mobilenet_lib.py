#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:37:51 2019

@author: ubuntu
"""
import torch.nn as nn


class ConvBNReLU():
    
    def __init__(self):
        pass
    
    def forward(self, x):
        pass


class Mobilenet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    
    def forward(self, x):
        pass


if __name__ == "__main__":
    import torchvision
#    model = torchvision.models.mobilenet_v2()
    model = torchvision.models.densenet161()
    print(model)
    